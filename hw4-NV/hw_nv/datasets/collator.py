import torch
import typing as tp
from torch.nn.utils.rnn import pad_sequence

from hw_nv.melspecs import MelSpectrogramConfig


def collate_fn(samples: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    transcripts = [it["transcript"] for it in samples]
    waveforms = pad_sequence(
        [it["waveform"] for it in samples],
        batch_first=True
    )

    waveforms_lengths = torch.cat(
        [it["waveform_length"] for it in samples],
        dim=0
    )

    melspecs = pad_sequence(
        [it["melspec"].transpose(-2, -1) for it in samples],
        batch_first=True,
        padding_value=MelSpectrogramConfig.pad_value
    ).transpose(-2, -1)

    melspecs_lengths = torch.cat(
        [it["melspec_length"] for it in samples],
        dim=0
    )

    return {
        "transcripts": transcripts,
        "waveforms": waveforms,
        "waveforms_lengths": waveforms_lengths,
        "melspecs": melspecs,
        "melspecs_lengths": melspecs_lengths
    }
