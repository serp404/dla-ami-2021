import os
import argparse
import warnings
import random
import wandb
import transformers

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from hw_nv.melspecs import MelSpectrogram, MelSpectrogramConfig
from hw_nv.datasets import prepare_dataloaders
from hw_nv.models import Generator
from hw_nv.models import MultiPeriodDiscriminator, MultiScaleDiscriminator
from hw_nv.utils import get_grad_norm, clip_gradients, traverse_config
from hw_nv.config import TaskConfig

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 3407
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def main(args):
    pargs = args.parse_args()
    resume_gen = pargs.resume_gen
    resume_dis = pargs.resume_dis
    device_opt = pargs.device

    if device_opt is not None:
        DEVICE = torch.device(device_opt)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_EPOCHS = TaskConfig.n_epochs
    BATCH_SIZE = TaskConfig.batch_size

    wandb.login()
    run = wandb.init(
        project="nv_project",
        entity="serp404",
        config=traverse_config(TaskConfig)
    )
    save_path = os.path.join(TaskConfig.save_dir, run.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    train_loader, val_loader = prepare_dataloaders(
        batch_size=BATCH_SIZE,
        train_size=TaskConfig.train_size,
        num_workers=TaskConfig.dataloader_workers
    )

    gen = Generator(**TaskConfig.gen_params).to(DEVICE)
    if resume_gen is not None:
        gen.load_state_dict(torch.load(resume_gen))

    dis = nn.ModuleList([
        MultiPeriodDiscriminator(**TaskConfig.mpd_params),
        MultiScaleDiscriminator(**TaskConfig.msd_params)
    ]).to(DEVICE)
    if resume_dis is not None:
        dis.load_state_dict(torch.load(resume_dis))

    optimizer_gen = getattr(torch.optim, TaskConfig.optimizer_gen)(
        gen.parameters(), **TaskConfig.optimizer_gen_params
    )

    if TaskConfig.scheduler_gen is not None:
        scheduler_type = TaskConfig.scheduler_gen
        if TaskConfig.scheduler_gen in dir(torch.optim.lr_scheduler):
            scheduler_gen = getattr(torch.optim.lr_scheduler, scheduler_type)(
                optimizer_gen, **TaskConfig.scheduler_gen_params
            )
        elif TaskConfig.scheduler_gen in dir(transformers.optimization):
            scheduler_gen = getattr(transformers.optimization, scheduler_type)(
                optimizer_gen, **TaskConfig.scheduler_gen_params
            )
        else:
            raise ModuleNotFoundError(f"Unknown scheduler '{scheduler_type}'")

    optimizer_dis = getattr(torch.optim, TaskConfig.optimizer_dis)(
        dis.parameters(), **TaskConfig.optimizer_dis_params
    )

    if TaskConfig.scheduler_dis is not None:
        scheduler_type = TaskConfig.scheduler_dis
        if TaskConfig.scheduler_dis in dir(torch.optim.lr_scheduler):
            scheduler_dis = getattr(torch.optim.lr_scheduler, scheduler_type)(
                optimizer_gen, **TaskConfig.scheduler_dis_params
            )
        elif TaskConfig.scheduler_dis in dir(transformers.optimization):
            scheduler_dis = getattr(transformers.optimization, scheduler_type)(
                optimizer_gen, **TaskConfig.scheduler_dis_params
            )
        else:
            raise ModuleNotFoundError(f"Unknown scheduler '{scheduler_type}'")

    mel_featurizer = MelSpectrogram(MelSpectrogramConfig())
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()

    for epoch in range(N_EPOCHS):
        torch.cuda.empty_cache()
        if TaskConfig.scheduler_gen is not None:
            scheduler_gen.step()

        if TaskConfig.scheduler_dis is not None:
            scheduler_dis.step()

        # Training loop
        gen.train()
        dis.train()
        train_losses_gen = []
        train_losses_dis = []
        train_grads_gen = []
        train_grads_dis = []
        for i, batch in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}"):
            if i > 1:
                break
            n_samples = batch["waveforms"].shape[0]
            mels_real = batch["melspecs"].to(DEVICE)
            wavs_real = batch["waveforms"].unsqueeze(dim=1).to(DEVICE)

            wavs_fake = gen(mels_real)
            mels_fake = mel_featurizer(
                wavs_fake.squeeze(dim=1).cpu()
            ).to(DEVICE)

            real_labels = torch.ones(n_samples, device=DEVICE)
            fake_labels = torch.zeros(n_samples, device=DEVICE)

            # discriminator
            loss_dis = 0.
            for d in dis:
                preds_real, _ = d(wavs_real)
                loss_dis += l2_criterion(preds_real, real_labels)
                preds_fake, _ = d(wavs_fake)
                loss_dis += l2_criterion(preds_fake.detach(), fake_labels)

            optimizer_dis.zero_grad()
            loss_dis.backward()
            clip_gradients(dis.parameters(), TaskConfig.grad_clip)
            train_grads_dis.append(get_grad_norm(dis))
            optimizer_dis.step()

            # generator
            max_len = min(mels_real.shape[-1], mels_fake.shape[-1])
            loss_gen = 45. * l1_criterion(
                mels_real[:, :, :max_len],
                mels_fake[:, :, :max_len]
            )
            for d in dis:
                preds_real, fmaps_real = d(wavs_real)
                preds_fake, fmaps_fake = d(wavs_fake)
                loss_gen += l2_criterion(preds_fake, real_labels)
                for fms_r, fms_f in zip(fmaps_real, fmaps_fake):
                    for fm_r, fm_f in zip(fms_r, fms_f):
                        loss_gen += 2. * l2_criterion(fm_r, fm_f)

            optimizer_gen.zero_grad()
            loss_gen.backward()
            clip_gradients(gen.parameters(), TaskConfig.grad_clip)
            train_grads_gen.append(get_grad_norm(gen))
            optimizer_gen.step()

            train_losses_dis.append(loss_dis.item())
            train_losses_gen.append(loss_gen.item())

        # Validation loop
        gen.eval()
        dis.eval()
        val_losses_gen = []
        val_losses_dis = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), desc=f"Validating epoch {epoch}"):
                if i > 1:
                    break
                n_samples = batch["waveforms"].shape[0]
                mels_real = batch["melspecs"].to(DEVICE)
                wavs_real = batch["waveforms"].unsqueeze(dim=1).to(DEVICE)

                wavs_fake = gen(mels_real)
                mels_fake = mel_featurizer(
                    wavs_fake.squeeze(dim=1).cpu()
                ).to(DEVICE)

                real_labels = torch.ones(n_samples, device=DEVICE)
                fake_labels = torch.zeros(n_samples, device=DEVICE)

                # discriminator
                loss_dis = 0.
                for d in dis:
                    preds_real, _ = d(wavs_real)
                    loss_dis += l2_criterion(preds_real, real_labels)
                    preds_fake, _ = d(wavs_fake)
                    loss_dis += l2_criterion(preds_fake, fake_labels)

                # generator
                max_len = min(mels_real.shape[-1], mels_fake.shape[-1])
                loss_gen = 45. * l1_criterion(
                    mels_real[:, :, :max_len],
                    mels_fake[:, :, :max_len]
                )
                for d in dis:
                    preds_real, fmaps_real = d(wavs_real)
                    preds_fake, fmaps_fake = d(wavs_fake)
                    loss_gen += l2_criterion(preds_fake, real_labels)
                    for fms_r, fms_f in zip(fmaps_real, fmaps_fake):
                        for fm_r, fm_f in zip(fms_r, fms_f):
                            loss_gen += 2. * l2_criterion(fm_r, fm_f)

                val_losses_dis.append(loss_dis.item())
                val_losses_gen.append(loss_gen.item())

        example_batch = next(iter(val_loader))
        with torch.no_grad():
            predicted_wavs = gen(example_batch).squeeze(dim=1)
        predicted_mels = mel_featurizer(wavs_fake.cpu()).to(DEVICE)

        audios = [
            wandb.Audio(
                predicted_wavs[i].cpu().numpy(),
                sample_rate=MelSpectrogramConfig.sr,
                caption=example_batch['transcripts'][i]
            ) for i in range(TaskConfig.examples_cnt)
        ]

        melspecs = [
            wandb.Image(
                predicted_mels[i].cpu().numpy(),
                caption=example_batch['transcripts'][i]
            ) for i in range(TaskConfig.examples_cnt)
        ]

        wandb.log(
            {
                "epoch": epoch,
                "train_loss_gen": np.mean(train_losses_gen),
                "train_loss_dis": np.mean(train_losses_dis),
                "grad_norm_gen": np.mean(train_grads_gen),
                "grad_norm_dis": np.mean(train_grads_dis),
                "lr_gen": optimizer_gen.param_groups[0]['lr'],
                "lr_dis": optimizer_dis.param_groups[0]['lr'],
                "val_audios": audios,
                "val_melspecs": melspecs
            }
        )

        if epoch % TaskConfig.save_period == 0:
            torch.save(
                gen.state_dict(),
                os.path.join(save_path, f"gen_e{epoch}.pth")
            )

            torch.save(
                dis.state_dict(),
                os.path.join(save_path, f"dis_e{epoch}.pth")
            )

    run.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-g",
        "--resume_gen",
        default=None,
        type=str,
        help="path to latest generator checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--resume_dis",
        default=None,
        type=str,
        help="path to latest discriminator checkpoint (default: None)",
    )
    args.add_argument(
        "-D",
        "--device",
        default=None,
        type=str,
        help="cpu of cuda (default: cuda if possible else cpu)",
    )

    main(args)
