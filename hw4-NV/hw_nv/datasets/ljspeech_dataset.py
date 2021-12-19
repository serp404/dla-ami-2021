import torchaudio

from hw_nv.melspecs import MelSpectrogram, MelSpectrogramConfig
from hw_nv.utils.utils import split_wav


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str, max_wav_len: int):
        super().__init__(root=root)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())
        self.processed_data = []
        for i in range(super().__len__()):
            waveform, _, _, transcript = super().__getitem__(i)
            self.processed_data.extend(
                split_wav(
                    wav=waveform,
                    text=transcript,
                    featurizer=self.featurizer,
                    max_len=max_wav_len
                )
            )

    def __getitem__(self, index: int):
        return self.processed_data[index]

    def __len__(self):
        return len(self.processed_data)
