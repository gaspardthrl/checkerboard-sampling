import numpy as np


def to_model_space(x: np.ndarray) -> np.ndarray:
    """[0,1]^D -> [-1,1]^D, matching the scale diffusion/flow-matching priors expect."""
    return x * 2 - 1


def from_model_space(x: np.ndarray) -> np.ndarray:
    """[-1,1]^D -> [0,1]^D, inverse of to_model_space."""
    return (x + 1) / 2
