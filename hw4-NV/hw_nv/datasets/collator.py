import typing as tp
from torch.nn.utils.rnn import pad_sequence

from hw_nv.utils.utils import split_wav
from hw_nv.melspecs import MelSpectrogram, MelSpectrogramConfig
from hw_nv.config import TaskConfig


featurizer = MelSpectrogram(MelSpectrogramConfig())
max_wav_len = TaskConfig.max_wav_len


def collate_fn(samples: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    transcripts = []
    waveforms = []
    melspecs = []

    for it in samples:
        for s in split_wav(
            wav=it["waveform"], text=it["transcript"],
            featurizer=featurizer, max_len=max_wav_len
        ):
            transcripts.append(s["transcript"])
            waveforms.append(s["waveform"])
            melspecs.append(s["melspec"])

    waveforms_tensor = pad_sequence(
        waveforms, batch_first=True
    )

    melspecs_tensor = pad_sequence(
        [it.transpose(-2, -1) for it in melspecs],
        batch_first=True,
        padding_value=MelSpectrogramConfig.pad_value
    ).transpose(-2, -1)

    return {
        "transcripts": transcripts,
        "waveforms": waveforms_tensor,
        "melspecs": melspecs_tensor
    }
