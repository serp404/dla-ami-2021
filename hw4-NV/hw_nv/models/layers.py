import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp


class MRFBlock(torch.nn.Module):
    def __init__(
        self, channels: int, kernel: int,
        dilations: tp.Tuple[tp.Tuple[int]],
        slope: float = 0.1
    ) -> None:

        super().__init__()

        n_layers = len(dilations)
        self.convnet1 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(negative_slope=slope),
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel, dilation=dilations[i][0],
                    padding="same"
                )
            ) for i in range(n_layers)
        ])

        self.convnet2 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(negative_slope=slope),
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel, dilation=dilations[i][1],
                    padding="same"
                )
            ) for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block1, block2 in zip(self.convnet1, self.convnet2):
            xr = block2(block1(x))
            x = x + xr
        return x


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, slope=0.1):
        super().__init__()
        self.period = period
        self.main_layers = nn.ModuleList([
            *sum(
                [
                    [
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=1 if i == 0 else 2**(5 + i),
                                out_channels=2**(6 + i),
                                kernel_size=(5, 1),
                                stride=(3, 1),
                                padding=2
                            ),
                            nn.LeakyReLU(negative_slope=slope)
                        )
                    ] for i in range(4)
                ],
                start=[]
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=512, out_channels=1024,
                    kernel_size=(5, 1), padding="same"
                ),
                nn.LeakyReLU(negative_slope=slope)
            )
        ])

        self.output_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=1,
                kernel_size=(3, 1), padding="same"
            ),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        batch_size, n_channels, seq_len = x.shape
        pad_size = self.period - (seq_len % self.period)
        x = F.pad(x, (0, pad_size), mode="reflect")
        seq_len += pad_size

        x = x.view(batch_size, n_channels, seq_len // self.period, self.period)
        fmaps = []
        for block in self.main_layers:
            x = block(x)
            fmaps.append(x)

        return self.output_layers(x), fmaps


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, slope=0.1):
        super().__init__()

        self.main_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 128, 15, 1, padding=7),
                nn.LeakyReLU(negative_slope=slope)
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
                nn.LeakyReLU(negative_slope=slope),
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=slope)
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=slope)
            ),
            nn.Sequential(
                nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=slope)
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=slope),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, 5, 1, padding=2),
                nn.LeakyReLU(negative_slope=slope)
            )
        ])

        self.output_layers = nn.Sequential(
            nn.Conv1d(1024, 1, 3, 1, padding=1),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        fmaps = []
        for block in self.main_layers:
            x = block(x)
            fmaps.append(x)
        return self.output_layers(x), fmaps
