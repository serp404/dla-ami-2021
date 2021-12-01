import torch
import torch.nn as nn
import typing as tp
from torch.nn.utils.rnn import pad_sequence

from hw_tts.models.positional_encodings import PositionalEncoding
from hw_tts.models.layers import FeedForwardTransformer, DurationPredictor


class FastSpeech(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, n_tokens: int, hidden_size: int,
        duration_hidden: int, n_encoders: int, n_decoders: int,
        alpha: float = 1., melspec_size: int = 80, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.embedding = nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=d_model,
            padding_idx=0
        )

        self.pos_encoding = PositionalEncoding(
            emb_size=d_model,
            maxlen=500000
        )

        self.encoder_layers = nn.Sequential(
            *[FeedForwardTransformer(n_head, d_model, hidden_size, kernel_size)
                for _ in range(n_encoders)]
        )

        self.duration_predictor = DurationPredictor(
            d_model, duration_hidden, kernel_size
        )

        self.decoder_layers = nn.Sequential(
            *[FeedForwardTransformer(n_head, d_model, hidden_size, kernel_size)
                for _ in range(n_decoders)],
            nn.Linear(in_features=d_model, out_features=melspec_size)
        )

    def forward(self, batch: tp.Dict[str, tp.Any]) -> torch.Tensor:
        device = next(self.parameters()).device

        tokens = batch["tokens"].to(device)
        tokens_lengths = batch["tokens_lengths"].to(device)

        has_durations = "durations" in batch and batch["durations"] is not None
        durations = batch["durations"].to(device) if has_durations else None

        x = self.embedding(tokens)
        x += self.pos_encoding(x)

        encoder_states = self.encoder_layers(x)
        durations_predicted = self.duration_predictor(encoder_states).squeeze()

        if durations is None:
            durations = torch.maximum(
                durations_predicted * self.alpha,
                torch.ones_like(durations_predicted)
            ).long()
        else:
            durations = (durations * self.alpha).long()

        flattend_states = []
        for seq, seq_len, d in zip(encoder_states, tokens_lengths, durations):
            states = torch.repeat_interleave(seq[:seq_len], d[:seq_len], dim=0)
            flattend_states.append(states)

        flattend_states = pad_sequence(flattend_states, batch_first=True)
        flattend_states += self.pos_encoding(flattend_states)

        return self.decoder_layers(flattend_states), durations_predicted
