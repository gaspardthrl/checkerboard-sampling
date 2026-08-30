"""Scoring for exploration runs (S-MEME, FE), all in model space [-1,1]^dim.

Two rules this module exists to enforce.

1. Entropy alone cannot grade an exploration method. On a bounded support the
   maximum entropy is log|Omega| and nothing can beat it, so a run that reports
   MORE than the target has not explored better -- it has leaked mass off the
   support. Every evaluator therefore returns `validity` alongside `entropy`,
   and a `tv_to_uniform` that is only small when both are right.

2. Histogram entropy is the wrong estimator here. A fixed grid over a curved
   support puts a large fraction of its occupied cells on the boundary, and the
   plug-in estimator is biased low by roughly (#occupied cells)/(2n). We use the
   Kozachenko-Leonenko k-NN estimator instead, which has no binning artefacts,
   and -- importantly -- we let it see points outside the support so leakage
   shows up as entropy ABOVE target rather than being silently clipped away.
"""

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

__all__ = [
    "knn_entropy",
    "CheckerboardEvaluator",
    "build_evaluator",
]


def knn_entropy(x: np.ndarray, k: int = 4) -> float:
    """Kozachenko-Leonenko differential entropy estimate, in nats.

        H = -psi(k) + psi(n) + log(c_d) + (d/n) sum_i log(r_i)

    with r_i the distance to the i-th point's k-th neighbour and c_d the volume
    of the unit d-ball. Duplicated points give r=0 and would send H to -inf, so
    those distances are floored.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    n, d = x.shape
    if n <= k:
        return float("nan")

    # query k+1 because the first neighbour returned is the point itself
    dist, _ = cKDTree(x).query(x, k=k + 1)
    r = np.maximum(dist[:, k], 1e-12)

    log_c_d = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
    return float(-digamma(k) + digamma(n) + log_c_d + d * np.log(r).mean())


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def temper(base_masses, areas, exponent: float) -> np.ndarray:
    """Masses of p ∝ p_base^exponent over a partition p_base is constant on.

    S-MEME's mirror-descent step is exactly tempering (Eq. 9 of the diffusion
    paper), and tempering a piecewise-constant density leaves the partition
    alone -- only the level per cell changes. So this is a closed-form
    prediction available at EVERY iteration, not just at convergence, which
    localises a sign or scale bug far faster than an entropy curve does.

    Cells with zero base mass stay at zero: tempering cannot create support.
    """
    base = np.asarray(base_masses, dtype=float)
    areas = np.asarray(areas, dtype=float)
    dens = np.divide(base, areas, out=np.zeros_like(base), where=areas > 0)
    w = np.where(base > 0, np.power(np.clip(dens, 1e-300, None), exponent) * areas, 0.0)
    return w / w.sum()


def _finite(x: np.ndarray):
    m = np.isfinite(x).all(axis=1)
    return x[m], float(m.mean())


# --------------------------------------------------------------------------- #
# Checkerboard                                                                 #
# --------------------------------------------------------------------------- #


class CheckerboardEvaluator:
    """Grades against uniform on the active (even-parity) tiles of [-1,1]^dim."""

    def __init__(self, n_tiles: int, dim: int = 2):
        if dim != 2:
            raise ValueError("CheckerboardEvaluator currently assumes dim == 2")
        self.n_tiles = n_tiles
        self.dim = dim
        i, j = np.indices((n_tiles, n_tiles))
        self.active = (((i + j) % 2) == 0).ravel()
        # half of [-1,1]^2 is active, for any n_tiles
        self.target_entropy = math.log(2.0)
        self.uniform_masses = self.active / self.active.sum()

    def tile_masses(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.full(self.n_tiles**2, np.nan)
        idx = np.clip(
            np.floor((x + 1) / 2 * self.n_tiles).astype(int), 0, self.n_tiles - 1
        )
        flat = idx[:, 0] * self.n_tiles + idx[:, 1]
        return np.bincount(flat, minlength=self.n_tiles**2) / max(len(x), 1)

    def on_support(self, x: np.ndarray) -> np.ndarray:
        inside = (np.abs(x) <= 1.0).all(axis=-1)
        idx = np.clip(
            np.floor((x + 1) / 2 * self.n_tiles).astype(int), 0, self.n_tiles - 1
        )
        return inside & (((idx[:, 0] + idx[:, 1]) % 2) == 0)

    @property
    def partition_areas(self) -> np.ndarray:
        return np.full(self.n_tiles**2, (2.0 / self.n_tiles) ** 2)

    def partition_masses(self, x: np.ndarray) -> np.ndarray:
        in_box = (np.abs(x) <= 1.0).all(axis=-1)
        return self.tile_masses(x[in_box])

    def tempering_prediction(self, exponent: float, base_masses) -> np.ndarray:
        """Tiles have equal area, so this reduces to base^exponent renormalised."""
        return temper(base_masses, self.partition_areas, exponent)

    def analytic_entropy(self, masses) -> float:
        w = np.asarray(masses, dtype=float)
        nz = w > 0
        a = self.partition_areas[nz]
        return float(-(w[nz] * np.log(w[nz] / a)).sum())

    def evaluate(self, x: np.ndarray) -> dict:
        x, frac_finite = _finite(np.asarray(x, dtype=float))
        keep = self.on_support(x) if len(x) else np.zeros(0, dtype=bool)
        in_box = (np.abs(x) <= 1.0).all(axis=-1) if len(x) else np.zeros(0, dtype=bool)
        w = self.tile_masses(x[in_box])
        return {
            "entropy": knn_entropy(x),
            "on_support_entropy": knn_entropy(x[keep])
            if keep.sum() > 50
            else float("nan"),
            "validity": float(keep.mean()) if len(x) else float("nan"),
            "frac_finite": frac_finite,
            "oob": float(1.0 - in_box.mean()) if len(x) else float("nan"),
            "tile_masses": w.round(6).tolist(),
            "tv_to_uniform": _tv(
                w * (in_box.mean() if len(x) else np.nan), self.uniform_masses
            ),
        }

    def describe(self) -> str:
        return (
            f"checkerboard n_tiles={self.n_tiles}: H*={self.target_entropy:.4f} "
            f"({self.active.sum()} active tiles)"
        )

    def headline(self, row: dict) -> str:
        diverged = (
            ""
            if row["frac_finite"] == 1.0
            else (f"  NON-FINITE (finite={row['frac_finite']:.6g})")
        )
        return (
            f"H={row['entropy']:+.4f} val={row['validity']:.4f} "
            f"oob={row['oob']:.4f} TV={row['tv_to_uniform']:.4f}{diverged}"
        )


def build_evaluator(n_tiles: int = 4, dim: int = 2):
    return CheckerboardEvaluator(n_tiles, dim)
