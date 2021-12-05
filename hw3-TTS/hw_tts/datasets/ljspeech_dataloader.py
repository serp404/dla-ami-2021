import torch

from torch.utils.data import DataLoader, random_split
from hw_tts.datasets import LJSpeechDataset, collate_fn


def prepare_dataloaders(batch_size, train_size=0.8, num_workers=1):
    dataset = LJSpeechDataset('./hw_tts/data/')
    total_len = len(dataset)

    train_len = int(train_size * total_len)
    val_len = total_len - train_len

    train_part, val_part = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_part, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_part, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
