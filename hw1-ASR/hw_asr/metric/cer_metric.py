import torch
from torch import Tensor
from typing import List

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
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(text_encoder, "ctc_beam_search"), "No beam search available for this encoder"
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []
        log_prob_lens = kwargs['log_probs_length'].cpu().tolist()

        for log_prob_l, log_prob_v, target_text in zip(log_prob_lens, log_probs.cpu(), text):
            pred_text = self.text_encoder.ctc_beam_search(
                log_prob_v[:log_prob_l, :].unsqueeze(dim=0)
            )
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
