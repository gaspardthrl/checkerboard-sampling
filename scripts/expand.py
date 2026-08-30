"""Run Flow Expander on a pre-trained checkerboard CFM model.

Stage 1 (scripts/train.py --framework cfm): fit a flow model to an imbalanced target.
Stage 2 (this script): expand it toward uniform-on-support using a verifier.

"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from frameworks.cfm import CFM
from frameworks.regularized_exploration.flowexpander import (
    FEConfig,
    flow_expander,
)
from frameworks.regularized_exploration.verifiers import (
    checkerboard_verifier,
)
from scripts.train import build_model
from utils.device import preferred_device
from utils.exploration_metrics import CheckerboardEvaluator

PLOT_LIM = 1.5

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--K", type=int, default=10)
    p.add_argument(
        "--gamma0",
        type=float,
        default=1.5,
        help="expansion step size at k=1 (see --gamma-schedule)",
    )
    p.add_argument(
        "--gamma-schedule",
        choices=["decay", "const"],
        default="decay",
        help="'decay' is gamma0/(1+3(k-1)); 'const' keeps the original "
        "paper gamma_k fixed across FE iterations",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=2.0,
        help="projection strength; 0 gives NSE (no projection)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="constant L-FE KL coefficient; 0 gives G-FE/NSE",
    )
    p.add_argument("--tau", type=float, default=4.0, help="verifier penalty slope")
    p.add_argument(
        "--dilate",
        type=float,
        default=0.0,
        help=">0 makes the verifier weak (Omega_v strictly contains Omega)",
    )
    p.add_argument(
        "--lambda-end",
        type=float,
        default=0.95,
        help="set lambda_t=0 at and after this time to avoid the terminal score singularity",
    )
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--t-min", type=float, default=0.05)
    p.add_argument("--solver-N", type=int, default=4)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-eval", type=int, default=20000)
    p.add_argument("--eval-ode-steps", type=int, default=50)
    p.add_argument(
        "--check-sde",
        action="store_true",
        help="verify the memoryless SDE reproduces the ODE marginals",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--tag", default=None, help="override the output subdirectory name")
    return p.parse_args()


def build_problem(targs, args):
    verifier = checkerboard_verifier(targs.n_tiles, tau=args.tau, dilate=args.dilate)
    return CheckerboardEvaluator(targs.n_tiles, targs.board_dim), verifier


def draw_panel(ax, x, title):
    ax.hist2d(
        x[:, 0],
        x[:, 1],
        bins=120,
        range=[[-PLOT_LIM, PLOT_LIM], [-PLOT_LIM, PLOT_LIM]],
        cmap="magma",
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_scatter_panel(ax, x, title, max_points=8_000):
    x = x[np.isfinite(x).all(axis=1)][:max_points]
    ax.scatter(x[:, 0], x[:, 1], s=1.5, alpha=0.28, linewidths=0, rasterized=True)
    ax.set_xlim(-PLOT_LIM, PLOT_LIM)
    ax.set_ylim(-PLOT_LIM, PLOT_LIM)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_checkerboard_verifier(verifier, n_tiles, outpath):
    """Plot the fixed reward field used by checkerboard projection.

    This is deliberately independent of a learned flow: it makes discontinuities
    or unexpected directions in ``grad log v`` visible before interpreting an
    FE run.
    """
    grid = np.linspace(-1.25, 1.25, 161, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    points = torch.from_numpy(np.column_stack([xx.ravel(), yy.ravel()]))
    with torch.no_grad():
        log_v = verifier["log_v"](points).numpy().reshape(xx.shape)
        hard = verifier["hard"](points).numpy().reshape(xx.shape)
    grad = verifier["grad_log_v"](points).numpy().reshape(*xx.shape, 2)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(
        log_v,
        extent=[grid[0], grid[-1], grid[0], grid[-1]],
        origin="lower",
        cmap="magma_r",
        interpolation="nearest",
    )
    ax.contour(xx, yy, hard.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color="white", lw=1.2))

    stride = 8
    arrow_grad = grad[::stride, ::stride]
    arrow_x, arrow_y = xx[::stride, ::stride], yy[::stride, ::stride]
    norm = np.linalg.norm(arrow_grad, axis=-1)
    u = np.divide(arrow_grad[..., 0], norm, out=np.zeros_like(norm), where=norm > 1e-8)
    v = np.divide(arrow_grad[..., 1], norm, out=np.zeros_like(norm), where=norm > 1e-8)
    ax.quiver(arrow_x, arrow_y, u, v, color="white", alpha=0.8, scale=25, width=0.003)
    ax.set(
        title=f"Checkerboard verifier: log v and normalized grad log v ({n_tiles}x{n_tiles})",
        xlabel="model-space x",
        ylabel="model-space y",
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, label="log v")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def draw_checkerboard_support(ax, n_tiles):
    """Draw the hard checkerboard support without using the verifier values."""
    width = 2.0 / n_tiles
    for row in range(n_tiles):
        for col in range(n_tiles):
            if (row + col) % 2 == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (-1 + col * width, -1 + row * width),
                        width,
                        width,
                        facecolor="0.9",
                        edgecolor="0.75",
                        lw=0.4,
                        zorder=0,
                    )
                )
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color="0.25", lw=1.0))


def draw_checkerboard_verifier_arrows(verifier, n_tiles, outpath):
    """Arrow-only view of the raw verifier gradient on a fixed grid."""
    grid = np.linspace(-1.25, 1.25, 25, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    points = torch.from_numpy(np.column_stack([xx.ravel(), yy.ravel()]))
    grad = verifier["grad_log_v"](points).numpy().reshape(*xx.shape, 2)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    draw_checkerboard_support(ax, n_tiles)
    # ``scale=30`` is fixed: a vector of norm tau=4 has visible length about .13.
    ax.quiver(
        xx,
        yy,
        grad[..., 0],
        grad[..., 1],
        angles="xy",
        scale_units="xy",
        scale=30,
        color="tab:red",
        width=0.0035,
        alpha=0.85,
    )
    ax.set(
        xlim=(-1.3, 1.3),
        ylim=(-1.3, 1.3),
        aspect="equal",
        title="Checkerboard verifier gradient (raw arrows; no heatmap)",
        xlabel="model-space x",
        ylabel="model-space y",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def draw_transport_maps(cfm, pretrained, expanded, n_tiles, num_steps, outpath):
    """Compare the two deterministic maps from the *same* Gaussian latents.

    The third panel is not a time trajectory: it is the paired endpoint change
    caused by Flow Expander, ``T_FE(z) - T_pre(z)``, for each common latent z.
    """
    latent = torch.randn(180, pretrained.dim)
    with torch.no_grad():
        before = (
            cfm.sample(pretrained, x0=latent.clone(), num_steps=num_steps).cpu().numpy()
        )
        after = (
            cfm.sample(expanded, x0=latent.clone(), num_steps=num_steps).cpu().numpy()
        )
    z = latent.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    panels = [
        (z, before, "Pretrained transport: z → Tpre(z)"),
        (z, after, "FE transport: z → TFE(z)"),
        (before, after, "FE shift: Tpre(z) → TFE(z)"),
    ]
    for ax, (start, end, title) in zip(axes, panels):
        draw_checkerboard_support(ax, n_tiles)
        delta = end - start
        ax.quiver(
            start[:, 0],
            start[:, 1],
            delta[:, 0],
            delta[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="0.35",
            alpha=0.28,
            width=0.0022,
            zorder=1,
        )
        ax.scatter(start[:, 0], start[:, 1], s=4, color="0.55", alpha=0.35, zorder=2)
        ax.scatter(end[:, 0], end[:, 1], s=6, color="tab:blue", alpha=0.65, zorder=3)
        ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal", title=title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    targs = Namespace(**ckpt["args"])
    if targs.framework != "cfm":
        raise ValueError(
            "expand.py expects a CFM run; use --framework cfm in scripts/train.py"
        )
    u_pre = build_model(targs)
    u_pre.load_state_dict(ckpt["model"])
    device = preferred_device()

    u_pre.to(device).eval()

    evaluator, verifier = build_problem(targs, args)
    cfm = CFM()
    cfg = FEConfig(
        num_steps=args.num_steps,
        t_min=args.t_min,
        N=args.solver_N,
        m=args.m,
        inner_steps=args.inner_steps,
        lr=args.lr,
    )

    tag = args.tag or ("nse" if args.eta == 0 else ("lfe" if args.alpha > 0 else "gfe"))
    outdir = args.outdir or (args.run / f"{tag}_g{args.gamma0:g}_e{args.eta:g}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(evaluator.describe())
    print(
        f"method={tag}  K={args.K} gamma0={args.gamma0} alpha={args.alpha} eta={args.eta} device={device}"
    )

    @torch.no_grad()
    def sample(model, n):
        return (
            cfm.sample(model, n=n, dim=model.dim, num_steps=args.eval_ode_steps)
            .cpu()
            .numpy()
        )

    def gamma_at(k):
        if args.gamma_schedule == "const":
            return args.gamma0
        return args.gamma0 / (1 + 3 * (k - 1))

    if args.check_sde:
        trajectory = cfm.sample(
            u_pre,
            n=8000,
            num_steps=cfg.num_steps,
            full=True,
            memoryless_sde=True,
            t_min=cfg.t_min,
        )
        a = evaluator.evaluate(trajectory.states[-1].cpu().numpy())
        b = evaluator.evaluate(sample(u_pre, 8000))
        print(
            f"SDE-vs-ODE  H {a['entropy']:+.4f} vs {b['entropy']:+.4f}   "
            f"validity {a['validity']:.4f} vs {b['validity']:.4f}  "
            f"(large gaps indicate coarse integration or imperfect CFM/SDE marginal matching)"
        )

    # ---- baseline ---------------------------------------------------------- #
    x0 = sample(u_pre, args.n_eval)
    rows = [{"k": 0, **evaluator.evaluate(x0)}]
    samples = {0: x0}
    print(f"\nk= 0  {evaluator.headline(rows[0])}")

    # ---- Flow Expander ----------------------------------------------------- #
    result = flow_expander(
        u_pre,
        [args.alpha] * args.K,
        [gamma_at(k) for k in range(1, args.K + 1)],
        [args.eta] * args.K,
        verifier,
        cfg,
        lambda_t=lambda t: float(t < args.lambda_end),
        return_history=True,
    )
    u_k = u_pre
    for k, item in enumerate(result.history, start=1):
        u_k = item["model"]
        xk = sample(u_k, args.n_eval)
        samples[k] = xk
        rows.append(
            {
                "k": k,
                **evaluator.evaluate(xk),
            }
        )
        print(f"k={k:2d}  {evaluator.headline(rows[-1])}")

    # ---- plots ------------------------------------------------------------- #
    target = evaluator.target_entropy
    ks = [r["k"] for r in rows]
    show = sorted({0} | {k for k in ks if k in samples})
    show = [k for k in show if k in samples]
    # Ensure we have at least one sample to plot if K=0
    if not show:
        show = [0]

    fig, axes = plt.subplots(1, len(show) + 3, figsize=(3.4 * (len(show) + 3), 3.5))
    for ax, k in zip(axes, show):
        draw_panel(ax, samples[k], "pre-trained" if k == 0 else f"{tag.upper()} k={k}")

    axes[-3].plot(ks, [r["entropy"] for r in rows], "s-", label="entropy")
    axes[-3].axhline(target, ls="--", c="k", label="log |support|")
    axes[-3].set_xlabel("iteration")
    axes[-3].set_ylabel("entropy (nats)")
    axes[-3].legend(fontsize=8)

    axes[-2].plot(ks, [r["validity"] for r in rows], "s-", c="tab:green")
    axes[-2].axhline(rows[0]["validity"], ls="--", c="k")
    axes[-2].set_ylim(-0.02, 1.05)
    axes[-2].set_xlabel("iteration")
    axes[-2].set_ylabel("validity")

    sc = axes[-1].scatter(
        [r["entropy"] for r in rows],
        [r["validity"] for r in rows],
        c=ks,
        cmap="viridis",
        zorder=3,
    )
    axes[-1].axvline(target, ls="--", c="k")
    axes[-1].set_ylim(-0.02, 1.05)
    axes[-1].set_xlabel("entropy")
    axes[-1].set_ylabel("validity")
    fig.colorbar(sc, ax=axes[-1], label="iteration")
    fig.tight_layout()
    fig.savefig(outdir / "expand.png", dpi=140)

    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, show):
        draw_scatter_panel(
            ax, samples[k], "pre-trained" if k == 0 else f"{tag.upper()} k={k}"
        )
    fig.tight_layout()
    fig.savefig(outdir / "expand_scatter.png", dpi=180)
    plt.close(fig)

    if verifier is not None:
        draw_checkerboard_verifier(
            verifier, targs.n_tiles, outdir / "verifier_field.png"
        )
        draw_checkerboard_verifier_arrows(
            verifier, targs.n_tiles, outdir / "verifier_arrows.png"
        )
        draw_transport_maps(
            cfm,
            u_pre,
            u_k,
            targs.n_tiles,
            args.eval_ode_steps,
            outdir / "transport_maps.png",
        )

    (outdir / "metrics.json").write_text(
        json.dumps(
            {
                "target_entropy": target,
                "method": tag,
                "dataset": getattr(targs, "dataset", "checkerboard"),
                "iterations": rows,
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
    )
    torch.save({"model": u_k.state_dict(), "args": ckpt["args"]}, outdir / "model.pt")
    print(f"\nwrote {outdir}/  (target entropy {target:.4f})")


if __name__ == "__main__":
    main()
