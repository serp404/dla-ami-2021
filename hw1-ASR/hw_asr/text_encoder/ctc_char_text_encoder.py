import os
import torch
import shutil
import wget
import gzip
from typing import List
from ctcdecode import CTCBeamDecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

# by tutorial:
# https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Offline_ASR.ipynb


def _get_kenlm():
    lm_gzip_path = '3-gram.pruned.1e-7.arpa.gz'
    if not os.path.exists(lm_gzip_path):
        print('Downloading pruned 3-gram model.')
        url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
        lm_gzip_path = wget.download(url)
        print('Downloaded the 3-gram language model.')
    else:
        print('Pruned .arpa.gz already exists.')

    uppercase_lm_path = '3-gram.pruned.1e-7.arpa'
    if not os.path.exists(uppercase_lm_path):
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(uppercase_lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        print('Unzipped the 3-gram language model.')
    else:
        print('Unzipped .arpa already exists.')

    lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    print('Converted language model file to lowercase.')
    return lm_path


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
            model_path=_get_kenlm(),
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

    def ctc_beam_search(self, log_probs: torch.tensor) -> str:
        """
        Performs beam search and returns a list of pairs
        (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        _, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)

        result, _, _, out_len = self.beam_search.decode(log_probs)

        best_beam = result[0][0][:out_len[0][0]]
        return "".join([self.ind2char[int(c)] for c in best_beam])
