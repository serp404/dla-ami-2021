import torch
from typing import List
from ctcdecode import CTCBeamDecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }

        for i, text in enumerate(alphabet, start=1):
            self.ind2char[i] = text

        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_size = 100

        self.beam_search = CTCBeamDecoder(
            [self.EMPTY_TOK] + alphabet,
            model_path=None,
            alpha=0.4,
            beam_width=self.beam_size,
            num_processes=2,
            blank_id=0,
            log_probs_input=True
        )

    def ctc_decode(self, inds: List[int]) -> str:
        res = []
        last_blank = False
        for ind in inds:
            if ind == self.char2ind[self.EMPTY_TOK]:
                last_blank = True
            else:
                if len(res) == 0 or last_blank or res[-1] != ind:
                    res.append(ind)
                last_blank = False
        return ''.join([self.ind2char[int(c)] for c in res])

    def ctc_beam_search(self, log_probs: torch.tensor) -> List[str]:
        """
        Performs beam search and returns a list of pairs
        (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 3
        batch_size, _, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)

        result, _, _, out_len = self.beam_search.decode(log_probs)

        output = []
        for i in range(batch_size):
            best_beam = result[i][0][:out_len[i][0]]
            output.append("".join([self.ind2char[int(c)] for c in best_beam]))
        return output
