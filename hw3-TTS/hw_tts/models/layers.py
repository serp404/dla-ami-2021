import math
import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dk: int, dv: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.dk = dk
        self.dv = dv

        self.initial_transforms = nn.ModuleDict([
            ["query", nn.Linear(in_features=d_model, out_features=dk)],
            ["key", nn.Linear(in_features=d_model, out_features=dk)],
            ["value", nn.Linear(in_features=d_model, out_features=dv)]
        ])

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:

        query = self.initial_transforms["query"](Q)
        key = self.initial_transforms["key"](K).transpose(1, 2)
        value = self.initial_transforms["value"](V)

        scores = torch.matmul(query, key) / math.sqrt(self.d_model)
        return torch.matmul(torch.softmax(scores, dim=-1), value)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int) -> None:
        super().__init__()
        assert d_model % n_head == 0, "n_head must be multiplier of d_model ."
        self.n_head = n_head
        self.d_model = d_model
        dk = dv = d_model // n_head

        self.norm = nn.LayerNorm(normalized_shape=d_model)
        self.initial_layers = nn.ModuleDict([
            t, nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model),
                nn.ReLU()
            )
        ] for t in ["query", "key", "value"])

        self.attention_heads = nn.ModuleList(
            [SelfAttentionBlock(d_model, dk, dv)
                for _ in range(n_head)]
        )

        self.output_layer = nn.Linear(
            in_features=n_head * dv,
            out_features=d_model
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        normbatch = self.norm(batch)
        query = self.initial_layers["query"](normbatch)
        key = self.initial_layers["key"](normbatch)
        value = self.initial_layers["value"](normbatch)

        heads_output = torch.cat(
            [head(query, key, value) for head in self.attention_heads],
            dim=-1
        )
        return self.output_layer(heads_output)


class ConvNet(nn.Module):
    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same"
            )
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_t = batch.transpose(1, 2)
        return self.layers(batch_t).transpose(1, 2)


class FeedForwardTransformer(nn.Module):
    def __init__(self, n_head: int, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.multihead = MultiHeadAttention(n_head, d_model)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.convs = ConvNet(d_model, kernel_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.multihead(batch)
        x = self.norm1(x + batch)
        x = self.convs(x)
        x = self.norm2(x + batch)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding="same"),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)

        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding="same"),
            nn.ReLU()
        )
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.regressor = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_t = batch.transpose(1, 2)
        x = self.conv1(batch_t).transpose(1, 2)
        x = self.norm1(x).transpose(1, 2)
        x = self.conv2(x).transpose(1, 2)
        return self.regressor(self.norm2(x))
