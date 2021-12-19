import torchaudio

from hw_nv.melspecs import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str):
        super().__init__(root=root)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)

        return {
            "transcript": transcript,
            "waveform": waveform
        }
