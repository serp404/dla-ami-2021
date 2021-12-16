import torch
import torch.nn as nn
import typing as tp

from hw_nv.models.layers import MRFBlock
from hw_nv.models.layers import PeriodDiscriminator, ScaleDiscriminator
from hw_nv.utils import init_weights
from hw_nv.utils.utils import normilize_weights


class Generator(torch.nn.Module):
    def __init__(
        self, channels_u: int, kernels_u: tp.Tuple[int],
        kernels_r: tp.Tuple[int], dilations_r: tp.Tuple[tp.Tuple[int]],
        slope: float = 0.1
    ) -> None:
        super().__init__()
        self.n_layers = len(kernels_u)

        self.initial_layers = nn.Conv1d(
            in_channels=80,
            out_channels=channels_u,
            kernel_size=7,
            padding="same"
        )

        self.main_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.main_layers.append(
                nn.Sequential(
                    nn.LeakyReLU(negative_slope=slope),
                    nn.ConvTranspose1d(
                        in_channels=channels_u // (2**i),
                        out_channels=channels_u // (2**(i+1)),
                        kernel_size=kernels_u[i],
                        stride=kernels_u[i] // 2,
                        padding=kernels_u[i] // 4
                    ),
                    MRFBlock(
                        channels=channels_u // (2**(i+1)),
                        kernel=kernels_r[i],
                        dilations=dilations_r,
                        slope=slope
                    )
                )
            )

        self.output_layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=slope),
            nn.Conv1d(
                in_channels=channels_u // (2**self.n_layers),
                out_channels=1,
                kernel_size=7,
                padding="same"
            ),
            nn.Tanh()
        )

        self.apply(init_weights)
        self.apply(normilize_weights)

    def forward(self, x):
        x = self.initial_layers(x)
        for m in self.main_layers:
            x = m(x)
        return self.output_layers(x)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period=p) for p in periods
        ])

    def forward(self, y):
        return torch.cat(
            [d(y) for d in self.discriminators],
            dim=-1
        )


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])

        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_siz=2, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=4, padding=4)
        ])

    def forward(self, y):
        return torch.cat(
            [d(pool(y)) for pool, d in zip(self.pools, self.discriminators)],
            dim=-1
        )
