from unittest.mock import patch
import pytest
import torch

from tinyvllm.utils.device import (
    _to_torch_dtype,
    is_triton_available,
    is_triton_interpreter,
    pick_device,
    pick_dtype,
)


def test_pick_device_defaults_to_cpu_when_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        dev = pick_device()
        assert dev.type == "cpu"


def test_pick_device_picks_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        dev = pick_device()
        assert dev.type == "cuda"


def test_pick_device_explicit():
    assert pick_device("cpu") == torch.device("cpu")
    assert pick_device(torch.device("cpu")) == torch.device("cpu")


def test_to_torch_dtype_resolution():
    assert _to_torch_dtype("float32") == torch.float32
    assert _to_torch_dtype("torch.float16") == torch.float16
    assert _to_torch_dtype("bfloat16") == torch.bfloat16
    assert _to_torch_dtype(torch.float32) == torch.float32
    assert _to_torch_dtype(None, default=torch.float16) == torch.float16

    with pytest.raises(ValueError):
        _to_torch_dtype("not_a_real_dtype")


def test_pick_dtype_cpu():
    cpu_dev = torch.device("cpu")
    assert pick_dtype(cpu_dev) == torch.float32
    assert pick_dtype(cpu_dev, "float16") == torch.float16


def test_pick_dtype_cuda_policies():
    cuda_dev = torch.device("cuda")

    # On modern GPUs (A100, L4) with BF16 support:
    with patch("torch.cuda.is_bf16_supported", return_value=True):
        assert pick_dtype(cuda_dev) == torch.bfloat16

    # On older GPUs (T4 sm_75) without BF16 compute support:
    with patch("torch.cuda.is_bf16_supported", return_value=False):
        assert pick_dtype(cuda_dev) == torch.float16


def test_is_triton_available():
    # Triton is installed in dev environment
    assert is_triton_available() is True


def test_is_triton_interpreter(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    assert is_triton_interpreter() is False

    monkeypatch.setenv("TRITON_INTERPRET", "1")
    assert is_triton_interpreter() is True

    monkeypatch.setenv("TRITON_INTERPRET", "0")
    assert is_triton_interpreter() is False
