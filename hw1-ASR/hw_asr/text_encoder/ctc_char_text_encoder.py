import torch
from typing import List, Tuple, Dict
from collections import defaultdict
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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

    def _extend_and_merge(self, next_probs: torch.tensor, src_paths: Dict[Tuple[str, str], float]):
        new_paths = defaultdict(float)
        for next_char_ind, next_char_prob in enumerate(next_probs):
            next_char = self.ind2char[next_char_ind]
            for (text, last_char), path_prob in src_paths.items():
                new_prefix = text if next_char == last_char else (text + next_char)
                new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
        return new_paths

    def _truncate_beam(self, paths, beam_size):
        return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs
        (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        paths = {('', self.EMPTY_TOK): 1.}
        for next_char_probs in probs:
            paths = self._extend_and_merge(next_char_probs, paths)
            paths = self._truncate_beam(paths, beam_size)
        # return sorted(hypos, key=lambda x: x[1], reverse=True)
        return [(prefix, score) for (prefix, _), score in sorted(paths.items(), key=lambda x: -x[1])]
