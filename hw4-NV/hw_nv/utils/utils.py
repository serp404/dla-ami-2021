import torch


def init_normal_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def normilize_simple_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.utils.weight_norm(m)


def normilize_spectral_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.utils.spectral_norm(m)


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


def clip_gradients(params, clip_value):
    if clip_value is not None:
        torch.nn.utils.clip_grad_norm_(params, clip_value)


def traverse_config(config):
    return {
        attr: getattr(config, attr) for attr in dir(config)
        if not attr.startswith("__")
    }


def split_wav(wav, text, featurizer, max_len):
    wav_parts = []
    for part in torch.split(wav, max_len, dim=1):
        if part.shape[-1] >= max_len // 10:
            waveform_length = torch.tensor([part.shape[-1]])

            melspec = featurizer(part)
            melspec_length = torch.tensor([melspec.shape[-1]])

            wav_parts.append(
                {
                    "transcript": text,
                    "waveform": part.squeeze(dim=0),
                    "waveform_length": waveform_length.long(),
                    "melspec": melspec.squeeze(dim=0),
                    "melspec_length": melspec_length.long()
                }
            )
    return wav_parts
