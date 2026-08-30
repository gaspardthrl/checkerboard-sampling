import numpy as np


class Checkerboard:
    def __init__(self, dim: int, n_tiles: int, weights: np.ndarray = None):
        self.dim = dim
        self.n_tiles = n_tiles

        if weights is None:
            idx = np.indices((n_tiles,) * dim)
            weights = (idx.sum(axis=0) % 2 == 0).astype(float)

        weights = np.asarray(weights, dtype=float)
        self.probs = weights / weights.sum()

    def sample(self, n: int, rng: np.random.Generator):
        flat_idx = rng.choice(self.n_tiles**self.dim, size=n, p=self.probs.ravel())
        idx = np.unravel_index(flat_idx, self.probs.shape)
        cell_origin = np.stack(idx, axis=-1) / self.n_tiles

        offset = rng.random((n, self.dim)) / self.n_tiles
        return cell_origin + offset

    def tile_index(self, x: np.ndarray) -> np.ndarray:
        idx = np.floor(x * self.n_tiles).astype(int)
        return np.clip(idx, 0, self.n_tiles - 1)

    def density(self, x: np.ndarray) -> np.ndarray:
        idx = self.tile_index(x)
        return (
            self.probs[tuple(idx[:, d] for d in range(self.dim))]
            * self.n_tiles**self.dim
        )
