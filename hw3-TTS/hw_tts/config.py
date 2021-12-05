import typing as tp


class TaskConfig:
    # Train options
    n_epochs: int = 5
    batch_size: int = 16
    dataloader_workers: int = 4
    train_size: float = 0.8
    l1_weight: float = 1.
    l2_weight: float = 1.

    # Model options
    model_params: tp.Dict[str, tp.Any] = {
        "d_model": 384,
        "n_head": 2,
        "n_tokens": 51,
        "n_encoders": 6,
        "n_decoders": 6,
        "hidden_size": 1536,
        "duration_hidden": 256,
        "alpha": 1.,
        "melspec_size": 80,
        "kernel_size": 3
    }

    # Optimization options
    grad_clip: tp.Optional[float] = None
    optimizer: str = "Adam"
    optimizer_params: tp.Dict[str, tp.Any] = {
        "lr": 0.001,
        "weight_decay": 1e-5
    }
    scheduler: tp.Optional[str] = "CosineAnnealingLR"
    scheduler_params: tp.Dict[str, tp.Any] = {
        "T_max": 5,
        "eta_min": 1e-5
    }

    # Checkpoint
    save_period: int = 1
    save_dir: str = "./hw_tts/log/"
