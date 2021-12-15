import torch
import torchaudio

from hw_nv.melspecs import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str):
        super().__init__(root=root)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]])

        melspec = self.featurizer(waveform)
        melspec_length = torch.tensor([melspec.shape[-1]])

        return {
            "transcript": transcript,
            "waveform": waveform.squeeze(dim=0),
            "waveform_length": waveform_length.long(),
            "melspec": melspec.squeeze(dim=0),
            "melspec_length": melspec_length.long()
        }
