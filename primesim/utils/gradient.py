import torch


def gradient_checkpoint_forward(module, use_gradient_checkpointing, *args, **kwargs):
    """module(*args, **kwargs), recomputing its activations in backward if asked."""
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            module, *args, **kwargs, use_reentrant=False
        )
    return module(*args, **kwargs)
