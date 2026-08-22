import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _tile_grid_lines(n_tiles: int):
    ticks = np.linspace(0, 1, n_tiles + 1)
    xs, ys, zs = [], [], []
    for a in ticks:
        for b in ticks:
            xs += [0, 1, None]; ys += [a, a, None]; zs += [b, b, None]
            xs += [a, a, None]; ys += [0, 1, None]; zs += [b, b, None]
            xs += [a, a, None]; ys += [b, b, None]; zs += [0, 1, None]
    return xs, ys, zs


def plot_samples_3d(samples: np.ndarray, n_tiles: int):
    """Interactive 3D scatter of sample points with a tile-boundary grid overlay."""
    return plot_samples_3d_comparison({"samples": samples}, n_tiles)


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
