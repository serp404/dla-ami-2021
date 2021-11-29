import torch
import torchaudio
from torchaudio.pipelines import TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH as taco


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str):
        super().__init__(root=root)
        self._tokenizer = taco.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]])
        tokens, token_lengths = self._tokenizer(transcript)
        return {
            "waveform": waveform.squeeze(dim=0),
            "waveform_length": waveform_length.long(),
            "transcript": transcript,
            "tokens": tokens.squeeze(dim=0),
            "tokens_length": token_lengths.long()
        }

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
