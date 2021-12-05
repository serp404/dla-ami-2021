import torch


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()


def compute_durations(aligner, batch, device):
    aligner_output = aligner(
        batch["waveforms"].to(device),
        batch["waveforms_lengths"],
        batch["transcripts"]
    ).cpu()

    batch["durations"] = (
        aligner_output * batch["melspecs_lengths"].unsqueeze(dim=1)
    ).long()

    max_len = min(batch["durations"].shape[-1], batch["tokens"].shape[-1])

    batch["durations"] = batch["durations"][:, :max_len]
    batch["tokens"] = batch["tokens"][:, :max_len]


def clip_gradients(params, clip_value):
    if clip_value is not None:
        torch.nn.utils.clip_grad_norm_(params, clip_value)


def traverse_config(config):
    return {
        attr: getattr(config, attr) for attr in dir(config)
        if not attr.startswith("__")
    }
