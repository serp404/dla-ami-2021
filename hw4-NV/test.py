import os
import argparse
import warnings
import random
import numpy as np

import torch
import torchaudio

from hw_nv.melspecs import MelSpectrogram, MelSpectrogramConfig
from hw_nv.models import Generator
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
    modelpath_opt = pargs.modelpath
    testpath_opt = pargs.testpath
    outputdir_opt = pargs.outputdir
    device_opt = pargs.device

    assert testpath_opt is not None, "Please, define test wavs dir."
    assert modelpath_opt is not None, "You must define pretrained generator."
    outputdir_opt = "./" if outputdir_opt is None else outputdir_opt

    if device_opt is not None:
        DEVICE = torch.device(device_opt)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = Generator(**TaskConfig.gen_params).to(DEVICE)
    gen.load_state_dict(torch.load(modelpath_opt, map_location=DEVICE))
    gen.eval()

    test_mels = []
    mel_featurizer = MelSpectrogram(MelSpectrogramConfig())
    for file in sorted(os.listdir(testpath_opt)):
        if file.endswith(".wav"):
            wav, _ = torchaudio.load(os.path.join(testpath_opt, file))
            test_mels.append(mel_featurizer(wav))

    predicted_wavs = []
    with torch.no_grad():
        for mels in test_mels:
            predicted_wavs.append(gen(mels.to(DEVICE)).cpu())

    if not os.path.exists(outputdir_opt):
        os.mkdir(outputdir_opt)

    for i, waveform in enumerate(predicted_wavs):
        path = os.path.join(outputdir_opt, f"test_wav_{i}.wav")
        torchaudio.save(
            path, waveform.unsqueeze(dim=0),
            MelSpectrogramConfig.sr
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-p",
        "--modelpath",
        default=None,
        type=str,
        help="path to tested generator",
    )
    args.add_argument(
        "-t",
        "--testpath",
        default=None,
        type=str,
        help="path to file with test waveforms",
    )
    args.add_argument(
        "-o",
        "--outputdir",
        default=None,
        type=str,
        help="dir to save output waveforms",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="cpu of cuda (default: cuda if possible else cpu)",
    )

    main(args)
