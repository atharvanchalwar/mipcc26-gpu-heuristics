# src/utils/device.py
"""
Device utilities for GPU/CPU selection.
"""

import torch


def get_default_device() -> torch.device:
    """
    Returns CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(array, device: torch.device):
    """
    Move a NumPy array or torch.Tensor to the given device.
    """
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array.to(device)
    return torch.as_tensor(array, device=device)
