"""Runtime device selection shared by command entrypoints."""

import torch


def preferred_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
