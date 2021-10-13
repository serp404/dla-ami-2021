import logging
import torch
import torch.nn as nn
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    audio_lst = [item['audio'].squeeze() for item in dataset_items]
    spectr_lst = [item['spectrogram'].squeeze().permute(1, 0) for item in dataset_items]
    enctext_lst = [item['text_encoded'].squeeze() for item in dataset_items]
    duration_lst = [item['duration'] for item in dataset_items]
    text_lst = [item['text'] for item in dataset_items]
    audio_path_lst = [item['audio_path'] for item in dataset_items]

    audio_tensor_pad = nn.utils.rnn.pad_sequence(audio_lst, batch_first=True, padding_value=0.)
    spectr_tensor_pad = nn.utils.rnn.pad_sequence(spectr_lst, batch_first=True, padding_value=0.)
    enctext_tensor_pad = nn.utils.rnn.pad_sequence(enctext_lst, batch_first=True, padding_value=0.)

    text_encoded_len = torch.as_tensor(list(map(lambda x: x['text_encoded'].shape[-1], dataset_items)))
    spectrogram_len = torch.as_tensor(list(map(lambda x: x['spectrogram'].shape[-1], dataset_items)))

    result_batch = {
        "audio": audio_tensor_pad,
        "spectrogram": spectr_tensor_pad.permute(0, 2, 1),
        "text_encoded": enctext_tensor_pad,
        "duration": duration_lst,
        "text": text_lst,
        "audio_path": audio_path_lst,
        "text_encoded_length": text_encoded_len,
        "spectrogram_length": spectrogram_len
    }
    return result_batch
