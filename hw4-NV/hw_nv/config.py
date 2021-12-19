import typing as tp


class TaskConfig:
    # Train options
    n_epochs: int = 2
    max_wav_len: int = 8192
    batch_size: int = 2
    dataloader_workers: int = 1
    train_size: float = 0.8

    # Model options
    gen_params: tp.Dict[str, tp.Any] = {
        "channels_u": 256,
        "kernels_u": (16, 16, 8),
        "kernels_r": (3, 5, 7),
        "dilations_r": ((1, 2), (2, 6), (3, 12)),
        "slope": 0.1
    }

    mpd_params: tp.Dict[str, tp.Any] = {}
    msd_params: tp.Dict[str, tp.Any] = {}

    # Optimization options
    grad_clip: tp.Optional[float] = 5.
    optimizer_gen: str = "AdamW"
    optimizer_gen_params: tp.Dict[str, tp.Any] = {
        "lr": 0.0002,
        "betas": (0.8, 0.99),
        "weight_decay": 0.01
    }
    scheduler_gen: tp.Optional[str] = "ExponentialLR"
    scheduler_gen_params: tp.Dict[str, tp.Any] = {"gamma": 0.999}

    optimizer_dis: str = "AdamW"
    optimizer_dis_params: tp.Dict[str, tp.Any] = {
        "lr": 0.0002,
        "betas": (0.8, 0.99),
        "weight_decay": 0.01
    }
    scheduler_dis: tp.Optional[str] = "ExponentialLR"
    scheduler_dis_params: tp.Dict[str, tp.Any] = {"gamma": 0.999}

    # Checkpoint
    save_period: int = 1
    save_dir: str = "./hw_nv/log/"
    examples_cnt: int = 1
