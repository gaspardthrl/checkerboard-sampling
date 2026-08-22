import numpy as np
import matplotlib.pyplot as plt


def _tile_occupancy(samples: np.ndarray, n_tiles: int) -> np.ndarray:
    idx = np.clip((samples * n_tiles).astype(int), 0, n_tiles - 1)
    counts = np.zeros((n_tiles, n_tiles, n_tiles))
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    return counts / counts.sum()


def plot_weight_slices(probs: np.ndarray):
    """2D heatmap of a tile weight/probability grid, one panel per z layer."""
    n_tiles = probs.shape[0]
    vmax = probs.max()
    fig, axes = plt.subplots(1, n_tiles, figsize=(3.2 * n_tiles + 1, 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        im = ax.imshow(probs[:, :, k].T, origin="lower", vmin=0, vmax=vmax, cmap="Blues")
        ax.set_title(f"z tile = {k}")
        ax.set_xlabel("x tile (i)")
        ax.set_ylabel("y tile (j)")
        ax.set_xticks(range(n_tiles))
        ax.set_yticks(range(n_tiles))

    fig.colorbar(im, ax=axes.tolist(), label="tile probability mass", shrink=0.8)
    return fig


def plot_tile_occupancy_comparison(datasets: dict[str, np.ndarray], n_tiles: int):
    """Grid of per-z-slice heatmaps: rows = dataset, columns = z tile.

    Empirical tile occupancy, computed by binning raw samples — works on
    ground-truth samples and generated samples alike, with a shared color
    scale across all panels so magnitudes are comparable.
    """
    labels = list(datasets.keys())
    occupancies = {label: _tile_occupancy(samples, n_tiles) for label, samples in datasets.items()}
    vmax = max(occ.max() for occ in occupancies.values())

    fig, axes = plt.subplots(
        len(labels), n_tiles,
        figsize=(3.2 * n_tiles + 1, 3.2 * len(labels)),
        constrained_layout=True, squeeze=False,
    )

    for row, label in enumerate(labels):
        occ = occupancies[label]
        for k in range(n_tiles):
            ax = axes[row, k]
            im = ax.imshow(occ[:, :, k].T, origin="lower", vmin=0, vmax=vmax, cmap="Blues")
            ax.set_title(f"{label} — z tile = {k}")
            ax.set_xlabel("x tile (i)")
            ax.set_ylabel("y tile (j)")
            ax.set_xticks(range(n_tiles))
            ax.set_yticks(range(n_tiles))

    fig.colorbar(im, ax=axes.ravel().tolist(), label="empirical tile probability mass", shrink=0.6)
    return fig
