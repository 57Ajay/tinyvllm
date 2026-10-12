import os
import importlib.util
import torch


def pick_device(user_device: torch.device | str | None = None) -> torch.device:
    if user_device is not None:
        return torch.device(user_device)
    else:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )


def _to_torch_dtype(
    dtype: str | torch.dtype | None, default: torch.dtype = torch.float32
) -> torch.dtype:
    if dtype is None:
        return default

    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str):
        # handle leading 'torch.', e.g. 'torch.float32'
        resolved = getattr(torch, dtype.split(".")[-1], None)
        if isinstance(resolved, torch.dtype):
            return resolved
        raise ValueError(f"'{dtype}' is not a valid PyTorch data type.")


def pick_dtype(
    device: torch.device, user_dtype: str | torch.dtype | None = None
) -> torch.dtype:
    if user_dtype is not None:
        return _to_torch_dtype(user_dtype)

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def is_triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def is_triton_interpreter() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"
