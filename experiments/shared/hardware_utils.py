import os
import torch
import platform
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Dataset


def detect_environment() -> str:
    """Detect execution environment: 'kaggle', 'colab', or 'local'."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def get_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def auto_batch_size(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    starting_batch: int = 32,
    safety_margin: int = 2,
) -> int:
    """OOM-safe dynamic batch size finder.
    
    Starts from `starting_batch` and halves until a safe size is found.
    """
    if not torch.cuda.is_available():
        return starting_batch

    model = model.cuda()
    model.train()
    batch_size = starting_batch

    while batch_size > 0:
        try:
            dummy = sample_input.unsqueeze(0).repeat(batch_size, *[1] * (sample_input.dim() - 1))
            dummy = dummy.cuda()
            output = model(dummy)
            loss = output.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return max(batch_size - safety_margin, 1)
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = max(batch_size // 2, 1)
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            else:
                raise e

    raise RuntimeError("Model too large to fit even a batch size of 1.")


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    if num_workers is None:
        num_workers = min(4, max(1, os.cpu_count() or 4) - 1)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )


def auto_config() -> dict:
    """Return a dict with recommended settings for the current environment."""
    env = detect_environment()
    vram = get_vram_gb()
    device = get_device()

    base = {
        "env": env,
        "device": str(device),
        "gpu_name": get_gpu_name(),
        "vram_gb": round(vram, 2),
        "amp": torch.cuda.is_available(),
        "gradient_accumulation": 1,
        "num_workers": min(4, max(1, (os.cpu_count() or 4) - 1)),
    }

    if env == "kaggle":
        base["gradient_accumulation"] = 1
    elif env == "colab":
        base["gradient_accumulation"] = 1
    else:
        if vram < 7.0:
            base["gradient_accumulation"] = 2
            base["amp"] = True

    return base


def auto_amp_scaler():
    if torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None


def hw_summary() -> str:
    env = detect_environment()
    device = get_device()
    gpu = get_gpu_name()
    vram = get_vram_gb()
    py = platform.python_version()
    torch_v = torch.__version__
    cuda_v = torch.version.cuda if torch.cuda.is_available() else "N/A"
    return (
        f"Environment: {env} | "
        f"Device: {device} | "
        f"GPU: {gpu} | "
        f"VRAM: {vram:.1f}GB | "
        f"PyTorch: {torch_v} | "
        f"CUDA: {cuda_v} | "
        f"Python: {py}"
    )
