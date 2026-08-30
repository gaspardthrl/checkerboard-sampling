"""Structure metrics the training loss does not capture.

Sampling can diverge and produce non-finite points. Those must never be scored:
casting NaN to an integer tile index yields tile 0, whose parity counts as a
checkerboard hit, so unfiltered NaN inflates active_frac instead of exposing it.
"""

import numpy as np


def finite_only(pts: np.ndarray):
    """Returns (finite_pts, frac_finite)."""
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask], float(mask.mean())


def checkerboard_metrics(pts: np.ndarray, n_tiles: int) -> dict:
    pts, frac_finite = finite_only(pts)
    if len(pts) == 0:
        return {"active_frac": float("nan"), "frac_finite": frac_finite,
                "frac_in_domain": float("nan")}

    in_domain = float(((pts >= 0) & (pts <= 1)).all(axis=1).mean())
    idx = np.clip((pts * n_tiles).astype(int), 0, n_tiles - 1)
    return {
        "active_frac": float(((idx.sum(axis=1) % 2) == 0).mean()),
        "frac_finite": frac_finite,
        "frac_in_domain": in_domain,
    }


def compute_metrics(pts: np.ndarray, n_tiles: int) -> dict:
    return checkerboard_metrics(pts, n_tiles)


def format_metrics(m: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())
