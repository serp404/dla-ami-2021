import torch
import torch.nn as nn

from hw_asr.base import BaseModel

# Inspired by https://github.com/SeanNaren/deepspeech.pytorch


class MaskConv(nn.Module):
    def __init__(self, modules_seq: nn.Sequential):
        super().__init__()
        self.modules_seq = modules_seq

    def forward(self, x, x_lengths):
        input_device = x.device

        for m in self.modules_seq:
            x = m(x)
            mask = torch.BoolTensor(x.shape).fill_(0).to(input_device)
            for i, length in enumerate(x_lengths):
                length = length.item()
                if mask[i].shape[2] - length > 0:
                    mask[i].narrow(
                        2, length,
                        mask[i].size(2) - length
                    ).fill_(1)
            x = x.masked_fill(mask, 0)
        return x


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.batch_norm_layer = nn.BatchNorm1d(num_features=input_size)
        self.rnn_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True
        )

    def forward(self, x, lengths):
        device = x.device
        seq_len, batch_size, _ = x.shape
        x = self.batch_norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(),
            enforce_sorted=False
        )

        initial_states = torch.randn(
            (2, batch_size, self.hidden_size),
            device=device
        )
        x, _ = self.rnn_layer(x, initial_states)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=seq_len)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(dim=2).view(
            x.shape[0], x.shape[1], -1
        )
        return x


class DeepSpeechModel(BaseModel):
    def __init__(
        self, n_feats, n_class, fc_hidden=256, rnn_hidden=512,
        num_layers=5, dropout=0.25, *args, **kwargs
    ):
        super().__init__(n_feats, n_class, *args, **kwargs)
        assert num_layers > 0, "Num layers must be greater than 0"

        self.conv_layers = MaskConv(
            nn.Sequential(
                nn.Conv2d(
                    1, 32, kernel_size=(41, 11),
                    stride=(2, 2), padding=(20, 5)
                ),
                nn.BatchNorm2d(num_features=32),
                nn.Hardtanh(min_value=0, max_value=20),
                nn.Conv2d(
                    32, 32, kernel_size=(21, 11),
                    stride=(2, 1), padding=(10, 5)
                ),
                nn.BatchNorm2d(num_features=32),
                nn.Hardtanh(min_value=0, max_value=20)
            )
        )

        rnn_input_size = (((n_feats - 1) // 2) // 2 + 1) * 32
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden
            )
        )

        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                BatchRNN(
                    input_size=rnn_hidden,
                    hidden_size=rnn_hidden
                )
            )

        self.fc_layers = nn.Sequential(
            nn.Linear(rnn_hidden, fc_hidden),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_class)
        )

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        x = spectrogram.unsqueeze(dim=1)
        lengths = spectrogram_length.cpu().int()

        output_lengths = self._get_seq_lens(lengths)
        x = self.conv_layers(x, output_lengths)
        x = x.view(x.shape[0], -1, x.shape[3]).permute(2, 0, 1)
        # seq_len, bs, dim

        for rnn in self.rnn_layers:
            x = rnn(x, output_lengths)

        logits = self.fc_layers(x).transpose(0, 1)
        return {"logits": logits}

    def _get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv_layers.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = (
                    seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1
                ) // m.stride[1] + 1
        return seq_len.int()

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
