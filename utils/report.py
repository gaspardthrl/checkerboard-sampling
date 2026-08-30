from utils.metrics import checkerboard_metrics, format_metrics
from viz.density import plot_tile_occupancy_comparison
from viz.scatter import (
    plot_samples_2d_comparison,
    plot_samples_2d_slices,
    plot_samples_3d_comparison,
)


def evaluate_and_plot(
    generated,
    ground_truth,
    board_dim,
    n_tiles,
    rundir,
    desc,
    label,
    suffix="",
    n_slices=None,
):
    """Print structure metrics and write comparison plots for one run.

    generated/ground_truth: (n, board_dim) numpy arrays in real [0,1]^board_dim space.
    desc: sampling-config text for the printed header, e.g. "1000 reverse steps".
    label: name for the generated-samples series in plot legends/titles.
    suffix: appended to output filenames, e.g. "_ddim" to avoid collisions between samplers.
    """
    metrics = checkerboard_metrics(generated, n_tiles)
    print(f"\n--- evaluation ({desc}) ---")
    print("  " + format_metrics(metrics))
    if metrics["frac_finite"] < 1.0:
        print(
            "  WARNING: sampling diverged for some points; excluded from metrics above"
        )

    datasets = {"true data": ground_truth, label: generated}
    if board_dim == 2:
        plot_samples_2d_comparison(datasets).savefig(
            rundir / f"scatter2d{suffix}.png", dpi=150
        )
    else:
        plot_samples_2d_slices(datasets, n_tiles, n_slices=n_slices).savefig(
            rundir / f"slices{suffix}.png", dpi=150
        )
        plot_samples_3d_comparison(datasets, n_tiles).write_html(
            rundir / f"scatter3d{suffix}.html"
        )
    plot_tile_occupancy_comparison(datasets, n_tiles).savefig(
        rundir / f"occupancy{suffix}.png", dpi=150
    )

    return metrics
