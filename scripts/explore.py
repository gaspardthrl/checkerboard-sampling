"""Run S-MEME on a bias-trained checkerboard DDPM.

Stage 1 (scripts/train.py): fit a DDPM to an imbalanced target, so p_pre is strongly
non-uniform over a known support.

Stage 2 (this script): run S-MEME and verify it flattens toward uniform.

Why this setting is a good test: the exact solution of Eq. (9) is tempering,

    p^k  proportional to  (p^{k-1})^(1 - 1/alpha)

and the target is piecewise constant on the known tile partition, so tempering leaves
the partition alone and only rescales the level on each cell. That gives a
closed-form prediction at *every* iteration, not just at convergence, which
localises bugs much faster than looking at an entropy curve.

S-MEME has no verifier, so nothing stops it leaking off the support. That is the
point of comparing it with Flow Expander: watch `validity`, not just entropy.

"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from frameworks.ddpm import DDPM
from frameworks.regularized_exploration.smeme import smeme
from scripts.train import build_model, build_scheduler
from utils.device import preferred_device
from utils.exploration_metrics import CheckerboardEvaluator

PLOT_LIM = 1.5


def draw_scatter_panel(ax, x, title, max_points=8_000):
    x = x[np.isfinite(x).all(axis=1)][:max_points]
    ax.scatter(x[:, 0], x[:, 1], s=1.5, alpha=0.28, linewidths=0, rasterized=True)
    ax.set_xlim(-PLOT_LIM, PLOT_LIM)
    ax.set_ylim(-PLOT_LIM, PLOT_LIM)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", type=Path, required=True, help="run dir containing model.pt")
    p.add_argument("--K", type=int, default=4, help="S-MEME iterations")
    p.add_argument(
        "--alpha", type=float, default=2.0,
        help="regularisation; exponent per step is (1-1/alpha). alpha=1 is the "
        "one-step Theorem 5.2 case, alpha>=1 required",
    )
    p.add_argument(
        "--score-sigma", type=float, default=0.4,
        help="noise scale at which the reward score is read",
    )
    p.add_argument("--reward-clip", type=float, default=10.0)
    p.add_argument("--N", type=int, default=30, help="rollouts per S-MEME iteration")
    p.add_argument("--m", type=int, default=64, help="trajectories per rollout")
    p.add_argument("--inner-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--n-eval", type=int, default=20000)
    p.add_argument("--eval-batch", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--tag", type=str, default=None, help="optional tag for output directory")
    return p.parse_args()


def build_problem(targs):
    return CheckerboardEvaluator(targs.n_tiles, targs.board_dim)


@torch.no_grad()
def sample_model(sampler, eps_model, n, batch):
    out = []
    for i in range(0, n, batch):
        out.append(sampler.sample(eps_model, min(batch, n - i)).cpu())
    return torch.cat(out).numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    targs = Namespace(**ckpt["args"])
    if targs.framework != "ddpm":
        raise ValueError("S-MEME here assumes the DDPM framework")
    device = preferred_device()
    scheduler = build_scheduler(targs)
    scheduler.to(device)
    eps_pre = build_model(targs, scheduler).to(device)
    eps_pre.load_state_dict(ckpt["model"])
    eps_pre.eval()

    evaluator = build_problem(targs)
    sampler = DDPM(scheduler)
    noise_levels = torch.sqrt(1.0 - scheduler.alpha_bars)
    score_index = int(torch.argmin((noise_levels - args.score_sigma).abs()).item())
    labels = getattr(scheduler, "timesteps", torch.arange(scheduler.T))
    score_timestep = int(labels[score_index])
    score_std = float(noise_levels[score_index])

    outdir = args.outdir or (args.run / f"smeme_a{args.alpha:g}_s{args.score_sigma:g}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(evaluator.describe())
    print(f"device={device}")
    print(f"reward score read at t={score_timestep} (sigma={score_std:.3f})")
    print(f"per-step tempering exponent: {1 - 1 / args.alpha:.4f}")

    # ---- baseline ---------------------------------------------------------- #
    x0 = sample_model(sampler, eps_pre, args.n_eval, args.eval_batch)
    has_tempering_prediction = hasattr(evaluator, "partition_masses")
    rows = [{"k": 0, **evaluator.evaluate(x0)}]
    if has_tempering_prediction:
        w0 = evaluator.partition_masses(x0)
        rows[0].update(
            partition_masses=w0.tolist(), predicted=w0.tolist(), tv_to_prediction=0.0
        )
    samples = {0: x0}
    print(f"\nk=0  {evaluator.headline(rows[0])}")

    # ---- S-MEME ------------------------------------------------------------ #
    result = smeme(
        args.K, eps_pre, [args.alpha] * args.K, scheduler,
        N=args.N, m=args.m, inner_steps=args.inner_steps, batch_size=args.batch_size,
        lr=args.lr, reward_clip=args.reward_clip, reward_noise_std=args.score_sigma,
        return_history=True,
    )
    for item in result.history:
        k, eps_k, log = item["k"] + 1, item["model"], item["log"]
        xk = sample_model(sampler, eps_k, args.n_eval, args.eval_batch)
        samples[k] = xk
        row = {
            "k": k,
            **evaluator.evaluate(xk),
            "mean_grad_f_norm": float(np.mean([r["grad_f_norm"] for r in log])),
        }
        if has_tempering_prediction:
            wk = evaluator.partition_masses(xk)
            # The exact prediction applies only when the base density is
            # piecewise constant on the evaluator's partition.
            pred = evaluator.tempering_prediction((1 - 1 / args.alpha) ** k, w0)
            row.update(
                partition_masses=wk.tolist(), predicted=pred.tolist(),
                tv_to_prediction=float(0.5 * np.abs(wk - pred).sum()),
            )
        rows.append(row)
        suffix = (f"  TV(sampled, tempering pred)={row['tv_to_prediction']:.4f}"
                  if has_tempering_prediction else "")
        print(f"k={k}  {evaluator.headline(row)}{suffix}  "
              f"|grad_f|={row['mean_grad_f_norm']:.3f}")

    # ---- plots ------------------------------------------------------------- #
    target = evaluator.target_entropy
    ks = [r["k"] for r in rows]

    fig, axes = plt.subplots(1, len(samples) + 2, figsize=(3.4 * (len(samples) + 2), 3.5))
    for ax, k in zip(axes, sorted(samples)):
        ax.hist2d(samples[k][:, 0], samples[k][:, 1], bins=120,
                  range=[[-PLOT_LIM, PLOT_LIM], [-PLOT_LIM, PLOT_LIM]], cmap="magma")
        ax.set_title("pre-trained" if k == 0 else f"S-MEME {k}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-2].plot(ks, [r["entropy"] for r in rows], "s-", label="S-MEME")
    axes[-2].axhline(target, ls="--", c="k", label="uniform on support")
    axes[-2].set_xlabel("iteration")
    axes[-2].set_ylabel("entropy (nats, model space)")
    axes[-2].legend(fontsize=8)

    axes[-1].plot(ks, [r["validity"] for r in rows], "s-", c="tab:green")
    axes[-1].set_ylim(-0.02, 1.05)
    axes[-1].set_xlabel("iteration")
    axes[-1].set_ylabel("validity (no verifier: expect leakage)")
    fig.tight_layout()
    fig.savefig(outdir / "smeme.png", dpi=140)

    fig, axes = plt.subplots(1, len(samples), figsize=(3.4 * len(samples), 3.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, sorted(samples)):
        draw_scatter_panel(ax, samples[k], "pre-trained" if k == 0 else f"S-MEME {k}")
    fig.tight_layout()
    fig.savefig(outdir / "smeme_scatter.png", dpi=180)
    plt.close(fig)

    (outdir / "metrics.json").write_text(json.dumps({
        "target_entropy": target,
        "dataset": getattr(targs, "dataset", "checkerboard"),
        "score_t": result.reward_timestep,
        "score_sigma": result.reward_noise_std,
        "alpha": args.alpha,
        "iterations": rows,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, indent=2))
    torch.save({"model": result.model.state_dict(), "args": ckpt["args"]}, outdir / "model.pt")
    print(f"\nwrote {outdir}/  (target entropy {target:.4f})")


if __name__ == "__main__":
    main()
