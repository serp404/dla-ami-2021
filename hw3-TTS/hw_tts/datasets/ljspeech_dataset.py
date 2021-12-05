import torch
import torchaudio
from torchaudio.pipelines import TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH as taco

from hw_tts.datasets.cleanup import basic_cleanup
from hw_tts.melspecs import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str):
        super().__init__(root=root)
        self._tokenizer = taco.get_text_processor()
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]])
        tokens, token_lengths = self._tokenizer(basic_cleanup(transcript))

        melspec = self.featurizer(waveform).transpose(-2, -1)
        melspec_length = torch.tensor([melspec.shape[-2]])

        return {
            "waveform": waveform.squeeze(dim=0),
            "waveform_length": waveform_length.long(),
            "transcript": transcript,
            "tokens": tokens.squeeze(dim=0),
            "tokens_length": token_lengths.long(),
            "melspec": melspec.squeeze(dim=0),
            "melspec_length": melspec_length.long()
        }
