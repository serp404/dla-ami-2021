import typing as tp


class TaskConfig:
    # Train options
    n_epochs: int = 60
    batch_size: int = 16
    dataloader_workers: int = 16
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
    grad_clip: tp.Optional[float] = 5.
    optimizer: str = "Adam"
    optimizer_params: tp.Dict[str, tp.Any] = {
        "lr": 0.0005,
        "betas": (0.9, 0.98),
        "eps": 1e-9
    }
    scheduler: tp.Optional[str] = "get_cosine_schedule_with_warmup"
    scheduler_params: tp.Dict[str, tp.Any] = {
        "num_warmup_steps": 6,
        "num_training_steps": 70
    }

    # Checkpoint
    save_period: int = 5
    save_dir: str = "./hw_tts/log/"
    examples_cnt: int = 5
