import numpy as np


class Checkerboard3D:
    def __init__(self, n_tiles: int, weights: np.ndarray = None):
        self.n_tiles = n_tiles

        if weights is None:
            i, j, k = np.indices((n_tiles, n_tiles, n_tiles))
            weights = ((i + j + k) % 2 == 0).astype(float)

        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_tiles, n_tiles, n_tiles):
            raise ValueError(
                f"weights must have shape ({n_tiles}, {n_tiles}, {n_tiles}), got {weights.shape}"
            )
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to a positive value")

        self.probs = weights / weights.sum()

    def sample(self, n: int, rng: np.random.Generator):
        flat_idx = rng.choice(self.n_tiles**3, size=n, p=self.probs.ravel())
        i, j, k = np.unravel_index(flat_idx, self.probs.shape)
        cell_origin = np.stack([i, j, k], axis=-1) / self.n_tiles

        offset = rng.random((n, 3)) / self.n_tiles
        return cell_origin + offset

    def tile_index(self, x: np.ndarray) -> np.ndarray:
        idx = np.floor(x * self.n_tiles).astype(int)
        return np.clip(idx, 0, self.n_tiles - 1)

    def density(self, x: np.ndarray) -> np.ndarray:
        idx = self.tile_index(x)
        return self.probs[idx[:, 0], idx[:, 1], idx[:, 2]] * self.n_tiles**3
