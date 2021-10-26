import torch
import torch.nn as nn
from hw_asr.base import BaseModel


class TransformerModel(BaseModel):
    def __init__(
        self, n_feats, n_class, fc_hidden=512, tr_hidden=256,
        num_layers=2, n_head=4, dropout=0.1, defualt_seq_len=1000,
        *args, **kwargs
    ):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.emb_size = n_feats
        self.fc_hidden = fc_hidden
        self.tr_hidden = tr_hidden
        self.n_class = n_class
        self.default_len = defualt_seq_len

        self.input_fc = nn.Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fc_hidden,
                nhead=n_head,
                dim_feedforward=tr_hidden,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        self.output_fc = nn.Sequential(
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

        self.register_buffer(
            "position_encodings",
            self._generate_pos_encodings(defualt_seq_len, fc_hidden)
        )

    def _generate_pos_encodings(self, seq_len, emb_size):
        ids = torch.arange(seq_len).unsqueeze(dim=1).expand((seq_len, emb_size))
        omegas = torch.pow(10000, -torch.arange(emb_size) // 2 / emb_size).unsqueeze(dim=0).expand((seq_len, emb_size))

        pos_encoding = torch.ones((seq_len, emb_size))
        pos_encoding[:, 0::2] = torch.sin(ids[:, 0::2] * omegas[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(ids[:, 1::2] * omegas[:, 1::2])
        return pos_encoding

    def forward(self, spectrogram, *args, **kwargs):
        model_device = next(self.parameters()).device
        batch = spectrogram.permute(0, 2, 1)

        seq_len = batch.shape[1]
        if seq_len <= self.default_len:
            pos_encodgins = self.position_encodings[:seq_len, :].to(model_device)
        else:
            pos_encodgins = self._generate_pos_encodings(
                seq_len,
                self.fc_hidden
            ).to(model_device)

        transformer_input = (self.input_fc(batch) + pos_encodgins).permute(1, 0, 2)
        enc_output = self.transformer(transformer_input)
        return {"logits": self.output_fc(enc_output.permute(1, 0, 2))}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
