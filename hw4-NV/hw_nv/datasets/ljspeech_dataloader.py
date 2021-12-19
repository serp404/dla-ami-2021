import torch

from torch.utils.data import DataLoader, random_split
from hw_nv.datasets.ljspeech_dataset import LJSpeechDataset
from hw_nv.datasets.sampler import GroupLengthBatchSampler
from hw_nv.datasets.collator import collate_fn


def prepare_dataloaders(
    batch_size, train_size=0.8,
    num_workers=1, length_sampler=True
):
    dataset = LJSpeechDataset('./hw_nv/data/')
    total_len = len(dataset)

    train_len = int(train_size * total_len)
    val_len = total_len - train_len

    train_part, val_part = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    if not length_sampler:
        train_loader = DataLoader(
            train_part, batch_size=batch_size, collate_fn=collate_fn,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_part, batch_sampler=GroupLengthBatchSampler(
                train_part, batch_size, batch_size * 8),
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )

    if not length_sampler:
        val_loader = DataLoader(
            val_part, batch_size=batch_size, collate_fn=collate_fn,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = DataLoader(
            val_part, batch_sampler=GroupLengthBatchSampler(
                val_part, batch_size, batch_size * 8),
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader
