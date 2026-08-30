import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_samples_2d_comparison(datasets: dict[str, np.ndarray], lim=(0, 1)):
    """Side-by-side 2D scatter, one panel per dataset, shared axes."""
    labels = list(datasets.keys())
    fig, axes = plt.subplots(
        1, len(labels), figsize=(4.5 * len(labels), 4.5), constrained_layout=True, squeeze=False
    )

    for ax, label in zip(axes[0], labels):
        samples = datasets[label]
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.4, edgecolors="none")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig


def plot_samples_2d_slices(
    datasets: dict[str, np.ndarray], n_tiles: int, n_slices: int = None, lim=(0, 1)
):
    """Scatter of (x, y) for the points falling in each z band.

    Rows = dataset, columns = z slab. Unlike the tile-occupancy heatmap this
    keeps individual points, so structure *within* a tile stays visible.
    """
    labels = list(datasets.keys())
    if n_slices is None:
        n_slices = n_tiles
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]
    edges = np.linspace(lim[0], lim[1], n_slices + 1)
    ticks = np.linspace(lim[0], lim[1], n_tiles + 1)

    fig, axes = plt.subplots(
        len(labels), n_slices,
        figsize=(3.0 * n_slices, 3.0 * len(labels)),
        constrained_layout=True, squeeze=False,
    )

    for row, (label, color) in enumerate(zip(labels, colors)):
        samples = datasets[label]
        samples = samples[np.isfinite(samples).all(axis=1)]
        z = samples[:, 2]
        for k in range(n_slices):
            # final slab includes the upper edge so z == lim[1] is not dropped
            upper = z <= edges[k + 1] if k == n_slices - 1 else z < edges[k + 1]
            pts = samples[(z >= edges[k]) & upper]

            ax = axes[row, k]
            for t in ticks:
                ax.axvline(t, color="lightgray", lw=0.6, zorder=0)
                ax.axhline(t, color="lightgray", lw=0.6, zorder=0)
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.45,
                       edgecolors="none", color=color, zorder=2)

            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal")
            ax.set_title(f"{label}\nz ∈ [{edges[k]:.2f}, {edges[k + 1]:.2f}]  n={len(pts)}",
                         fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    return fig


def _tile_grid_lines(n_tiles: int):
    ticks = np.linspace(0, 1, n_tiles + 1)
    xs, ys, zs = [], [], []
    for a in ticks:
        for b in ticks:
            xs += [0, 1, None]; ys += [a, a, None]; zs += [b, b, None]
            xs += [a, a, None]; ys += [0, 1, None]; zs += [b, b, None]
            xs += [a, a, None]; ys += [b, b, None]; zs += [0, 1, None]
    return xs, ys, zs


def plot_samples_2d_overlay(datasets: dict[str, np.ndarray], lim=(0, 1)):
    """Overlay 2D scatters on shared axes so distributions can be compared directly.

    Returns (fig, ax) so callers can annotate with reference geometry.
    """
    # blue/orange reads as two distinct series under the common CVD types
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for (label, samples), color in zip(datasets.items(), colors):
        ax.scatter(
            samples[:, 0], samples[:, 1],
            s=4, alpha=0.35, edgecolors="none", color=color, label=label,
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    legend = ax.legend(markerscale=3, framealpha=0.9)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return fig, ax


def plot_samples_3d_comparison(datasets: dict[str, np.ndarray], n_tiles: int):
    """Side-by-side interactive 3D scatter, one panel per dataset."""
    labels = list(datasets.keys())
    fig = make_subplots(
        rows=1, cols=len(labels),
        specs=[[{"type": "scene"}] * len(labels)],
        subplot_titles=labels,
    )
    grid_x, grid_y, grid_z = _tile_grid_lines(n_tiles)

    for col, label in enumerate(labels, start=1):
        samples = datasets[label]
        fig.add_trace(
            go.Scatter3d(
                x=grid_x, y=grid_y, z=grid_z,
                mode="lines", line=dict(color="lightgray", width=1),
                hoverinfo="skip", showlegend=False,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                mode="markers",
                marker=dict(size=2, color=samples[:, 2], colorscale="Viridis", opacity=0.8),
                showlegend=False,
            ),
            row=1, col=col,
        )

    scene_kwargs = dict(
        xaxis=dict(range=[0, 1], title="x"),
        yaxis=dict(range=[0, 1], title="y"),
        zaxis=dict(range=[0, 1], title="z"),
        aspectmode="cube",
    )
    scenes = {f"scene{'' if i == 1 else i}": scene_kwargs for i in range(1, len(labels) + 1)}
    fig.update_layout(**scenes, margin=dict(l=0, r=0, t=30, b=0))
    return fig
