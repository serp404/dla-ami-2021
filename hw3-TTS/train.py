import os
import sys
import argparse
import warnings
import random
import wandb
import transformers

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from hw_tts.aligners import GraphemeAligner
from hw_tts.vocoders import Vocoder
from hw_tts.melspecs import MelSpectrogramConfig
from hw_tts.datasets import prepare_dataloaders
from hw_tts.loss import fastspeech_loss
from hw_tts.models import FastSpeech
from hw_tts.config import TaskConfig
from hw_tts.utils import get_grad_norm, compute_durations
from hw_tts.utils import clip_gradients, traverse_config

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 3407
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def main(args):
    pargs = args.parse_args()
    resume_opt = pargs.resume
    device_opt = pargs.device

    if device_opt is not None:
        DEVICE = torch.device(device_opt)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_EPOCHS = TaskConfig.n_epochs
    BATCH_SIZE = TaskConfig.batch_size

    wandb.login()
    run = wandb.init(
        project="tts_project",
        entity="serp404",
        config=traverse_config(TaskConfig)
    )
    save_path = os.path.join(TaskConfig.save_dir, run.name)
    os.mkdir(save_path)

    train_loader, val_loader = prepare_dataloaders(
        batch_size=BATCH_SIZE,
        train_size=TaskConfig.train_size,
        num_workers=TaskConfig.dataloader_workers
    )

    model = FastSpeech(**TaskConfig.model_params)
    if resume_opt is not None:
        model.load_state_dict(torch.load(resume_opt))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(
                    p, gain=torch.nn.init.calculate_gain("linear")
                )

    model = model.to(DEVICE)
    aligner = GraphemeAligner(melspec_sr=MelSpectrogramConfig.sr).to(DEVICE)

    sys.path.append('waveglow/')
    vocoder = Vocoder(
        path="./hw_tts/data/waveglow_256channels_universal_v5.pt"
    ).to(DEVICE).eval()

    criterion = fastspeech_loss(
        l2_weight=TaskConfig.l2_weight,
        l1_weight=TaskConfig.l1_weight
    )

    optimizer = getattr(torch.optim, TaskConfig.optimizer)(
        model.parameters(), **TaskConfig.optimizer_params
    )

    if TaskConfig.scheduler is not None:
        scheduler_type = TaskConfig.scheduler
        if TaskConfig.scheduler in dir(torch.optim.lr_scheduler):
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
                optimizer, **TaskConfig.scheduler_params
            )
        elif TaskConfig.scheduler in dir(transformers.optimization):
            scheduler = getattr(transformers.optimization, scheduler_type)(
                optimizer, **TaskConfig.scheduler_params
            )
        else:
            raise ModuleNotFoundError(f"Unknown scheduler '{scheduler_type}'")

    for epoch in range(N_EPOCHS):
        if TaskConfig.scheduler is not None:
            scheduler.step()

        model.train()
        train_losses = []
        train_grads = []
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            compute_durations(aligner, batch, DEVICE)

            mels, durs = model(batch)
            max_mel_len = min(mels.shape[-2], batch["melspecs"].shape[-2])
            pad_tok_mask = (batch["tokens"] != 0).to(DEVICE)

            loss = criterion(
                mels_pred=mels[:, :max_mel_len],
                mels_true=batch["melspecs"][:, :max_mel_len].to(DEVICE),
                durs_pred=durs * pad_tok_mask,
                durs_true=torch.log1p(
                    batch["durations"].float().to(DEVICE)
                ) * pad_tok_mask
            )

            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            clip_gradients(model.parameters(), TaskConfig.grad_clip)
            train_grads.append(get_grad_norm(model))

            optimizer.step()

        model.eval()
        val_losses = []
        for batch in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
            compute_durations(aligner, batch, DEVICE)
            with torch.no_grad():
                mels, durs = model(batch)

            max_mel_len = min(mels.shape[-2], batch["melspecs"].shape[-2])
            pad_tok_mask = (batch["tokens"] != 0).to(DEVICE)

            loss = criterion(
                mels_pred=mels[:, :max_mel_len],
                mels_true=batch["melspecs"][:, :max_mel_len].to(DEVICE),
                durs_pred=durs * pad_tok_mask,
                durs_true=torch.log1p(
                    batch["durations"].float().to(DEVICE)
                ) * pad_tok_mask
            )

            val_losses.append(loss.item())

        example_batch = next(iter(val_loader))
        with torch.no_grad():
            predicted_mels, _ = model(example_batch)
        predicted_wavs = vocoder.inference(
            predicted_mels.transpose(1, 2).to(DEVICE)
        ).cpu()

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
                "train_loss": np.mean(train_losses),
                "val_loss": np.mean(val_losses),
                "grad_norm": np.mean(train_grads),
                "lr": optimizer.param_groups[0]['lr'],
                "val_audios": audios,
                "val_melspecs": melspecs
            }
        )

        if epoch % TaskConfig.save_period:
            torch.save(model.cpu(), os.path.join(save_path, f"e{epoch}"))

    run.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="cpu of cuda (default: cuda if possible else cpu)",
    )

    main(args)
