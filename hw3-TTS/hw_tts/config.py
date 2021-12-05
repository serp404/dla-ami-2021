import typing as tp


class TaskConfig:
    # Train options
    n_epochs: int = 100
    batch_size: int = 16
    dataloader_workers: int = 8
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
        "kernel_size": 3,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.2
    }

    # Optimization options
    grad_clip: tp.Optional[float] = 10.
    optimizer: str = "Adam"
    optimizer_params: tp.Dict[str, tp.Any] = {
        "lr": 0.0008,
        "betas": (0.9, 0.98),
        "eps": 1e-9
    }
    scheduler: tp.Optional[str] = "get_linear_schedule_with_warmup"
    scheduler_params: tp.Dict[str, tp.Any] = {
        "num_warmup_steps": 10,
        "num_training_steps": 100
    }

    # Checkpoint
    save_period: int = 5
    save_dir: str = "./hw_tts/log/"
    examples_cnt: int = 5
