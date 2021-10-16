import torch
import torch.nn as nn

from hw_asr.base import BaseModel


class CheckpointModel(BaseModel):
    def __init__(
        self, n_feats, n_class, fc_hidden=512, num_layers=1,
        rnn_hidden=128, bidirectional=False, *args, **kwargs
    ):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.fc_hidden = fc_hidden
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers
        self.bidirectioanl = bidirectional
        self.factor = 2 if bidirectional else 1

        self.rnn_net = nn.GRU(
            input_size=n_feats,
            hidden_size=rnn_hidden,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
            **kwargs
        )

        self.fc_net = nn.Sequential(
            nn.Linear(in_features=self.factor * rnn_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        batch_size = spectrogram.shape[0]
        model_device = next(self.parameters()).device
        initial_states = torch.randn(
            (self.factor * self.num_layers, batch_size, self.rnn_hidden),
            device=model_device
        )

        output, _ = self.rnn_net(spectrogram.permute(0, 2, 1), initial_states)
        return {"logits": self.fc_net(output)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
