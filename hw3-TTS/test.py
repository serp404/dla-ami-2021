import os
import sys
import argparse
import warnings
import random
import numpy as np

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.pipelines import TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH as taco

from hw_tts.datasets.cleanup import basic_cleanup
from hw_tts.melspecs import MelSpectrogramConfig
from hw_tts.vocoders import Vocoder
from hw_tts.models import FastSpeech
from hw_tts.config import TaskConfig

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

    assert testpath_opt is not None, "Please, define test sentences file."
    assert modelpath_opt is not None, "You must define pretrained model."
    outputdir_opt = "./" if outputdir_opt is None else outputdir_opt

    if device_opt is not None:
        DEVICE = torch.device(device_opt)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastSpeech(**TaskConfig.model_params).to(DEVICE)
    model = torch.load(modelpath_opt, map_location=DEVICE)
    model.eval()

    tokenizer = taco.get_text_processor()

    with open(testpath_opt, "r") as f:
        transcripts = [s.strip() for s in f.readlines()]

    tokens_list = []
    tokens_lengths_list = []
    for text in transcripts:
        tokens, tokens_lengths = tokenizer(basic_cleanup(text))
        tokens_list.append(tokens.squeeze(dim=0))
        tokens_lengths_list.append(tokens_lengths.long())

    test_batch = {
        "tokens": pad_sequence(tokens_list, batch_first=True),
        "tokens_lengths": torch.cat(tokens_lengths_list, dim=0),
        "transcripts": transcripts
    }

    sys.path.append('waveglow/')
    vocoder = Vocoder(
        path="./hw_tts/data/waveglow_256channels_universal_v5.pt"
    ).to(DEVICE).eval()

    with torch.no_grad():
        predicted_mels, _ = model(test_batch)

    predicted_wavs = vocoder.inference(
        predicted_mels.transpose(1, 2).to(DEVICE)
    ).cpu()

    for i, waveform in predicted_wavs:
        path = os.path.join(outputdir_opt, f"test_wav_{i}.wav")
        torchaudio.save(path, waveform, MelSpectrogramConfig.sr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-p",
        "--modelpath",
        default=None,
        type=str,
        help="path to tested model",
    )
    args.add_argument(
        "-t",
        "--testpath",
        default=None,
        type=str,
        help="path to file with converted semtences",
    )
    args.add_argument(
        "-o",
        "--outputdir",
        default=None,
        type=str,
        help="dir to save output wavs",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="cpu of cuda (default: cuda if possible else cpu)",
    )

    main(args)
