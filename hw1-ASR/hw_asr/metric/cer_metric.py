import torch
from torch import Tensor
from typing import List
from joblib import Parallel, delayed

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        for log_prob_vec, target_text in zip(predictions, text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(text_encoder, "ctc_beam_search"), "No beam search available for this encoder"
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []
        probs = torch.exp(log_probs).cpu()
        beam_search = self.text_encoder.ctc_beam_search
        predictions = Parallel(n_jobs=4)(
            delayed(beam_search)(distr, self.beam_size) for distr in probs
        )

        for pred_texts, target_text in zip(predictions, text):
            cers.append(calc_cer(target_text, pred_texts[0][0]))
        return sum(cers) / len(cers)
