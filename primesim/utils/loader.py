import torch
from safetensors import safe_open


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    """Read a .safetensors or torch .pth file, or a list of shards, into one dict."""
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype, device))
        return state_dict
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
    else:
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if torch_dtype is not None:
        state_dict = {
            k: v.to(torch_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }
    return state_dict
