import torch
import typing as tp
from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    transcripts = [it["transcript"] for it in samples]
    waveforms = pad_sequence(
        [it["waveform"] for it in samples]
    ).transpose(0, 1)
    waveforms_lengths = torch.cat(
        [it["waveform_length"] for it in samples],
        dim=0
    )

    tokens = pad_sequence(
        [it["tokens"] for it in samples]
    ).transpose(0, 1)
    tokens_lengths = torch.cat(
        [it["tokens_length"] for it in samples],
        dim=0
    )

    return {
        "waveforms": waveforms,
        "waveforms_lengths": waveforms_lengths,
        "transcripts": transcripts,
        "tokens": tokens,
        "tokens_lengths": tokens_lengths,
        "durations": None
    }
