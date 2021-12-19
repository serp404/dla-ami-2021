import torch
import random


class GroupLengthBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size, group_size):
        self.data = data_source
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.group_size = group_size
        self.id2group = torch.zeros(len(data_source))
        self.n_groups = len(data_source) // group_size

        argsort_ids = torch.argsort(
            torch.tensor([len(x['waveform']) for x in data_source])
        )

        for k in range(self.n_groups):
            lower_bound = k * self.group_size
            upper_bound = (k + 1) * self.group_size
            if k == self.n_groups - 1:
                upper_bound = self.data_len - 1
            ids = argsort_ids[lower_bound:upper_bound]
            self.id2group[ids] = k

    def __iter__(self):
        num_yielded = 0
        ids = torch.arange(self.data_len)
        while num_yielded < self.data_len:
            current_group = random.randint(0, self.n_groups - 1)
            current_group_size = torch.count_nonzero(
                self.id2group == current_group
            )

            random_ids = torch.randperm(current_group_size)[:self.batch_size]
            batch = ids[self.id2group == current_group][random_ids].tolist()
            num_yielded += self.batch_size
            yield batch

    def __len__(self):
        return self.data_len // self.batch_size
