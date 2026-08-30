"""Train a DDPM on the 3D checkerboard and compare its samples against ground truth.

Each run writes to outputs/runs/<name>/ (model.pt, metrics.json, plots), so
configurations never overwrite each other.

Sampling uses the same uniform time grid and the same T steps as training.

Examples:
    uv run python scripts/train.py --steps 100000
    uv run python scripts/train.py --steps 100000 --bias 0.75
    uv run python scripts/train.py --steps 100000 --timesteps 500 --name shortT
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.checkerboard import Checkerboard
from frameworks.cfm import CFM
from frameworks.ddpm import DDPM
from frameworks.scheduler import DDPMScheduler
from models.mlp import (
    MLP,
    FlowMLP,
    HeatGatedFourierFeatures,
)
from utils.normalize import from_model_space
from utils.device import preferred_device
from utils.report import evaluate_and_plot
from utils.trainer import train


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("data")
    g.add_argument("--framework", choices=["ddpm", "cfm"], default="ddpm")
    g.add_argument("--board-dim", type=int, default=3)
    g.add_argument("--n-tiles", type=int, default=4)
    g.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="exponential tile-weight decay away from tile (0,0,0)",
    )

    g = p.add_argument_group("model")
    g.add_argument(
        "--depth", type=int, default=4, help="number of FiLM residual hidden blocks"
    )
    g.add_argument("--hidden-dim", type=int, default=128)
    g.add_argument("--time-emb-dim", type=int, default=32)
    g.add_argument(
        "--num-features", type=int, default=256, help="random Fourier features"
    )
    g.add_argument(
        "--num-harmonics",
        type=float,
        default=16.0,
        help="RFF max frequency, as a multiple of the tile fundamental f0=N/4 "
        "(so headroom stays proportional to n-tiles)",
    )
    g.add_argument(
        "--no-rff",
        dest="use_rff",
        action="store_false",
        help="feed raw spatial coordinates instead of random Fourier features "
        "(reproduces the pre-fix spectral-bias failure mode)",
    )
    p.set_defaults(use_rff=True)

    g = p.add_argument_group("diffusion")
    g.add_argument("--timesteps", type=int, default=1000, help="T (ddpm only)")

    g = p.add_argument_group("flow")
    g.add_argument(
        "--ode-steps",
        type=int,
        default=50,
        help="ODE integration steps at sample time (cfm only)",
    )

    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=20000)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--log-every", type=int, default=1000)

    g = p.add_argument_group("output")
    g.add_argument(
        "--n-samples", type=int, default=20000, help="generated for evaluation"
    )
    g.add_argument(
        "--name", default=None, help="run name (default: derived from config)"
    )
    g.add_argument("--outdir", type=Path, default=Path("outputs/runs"))
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def default_run_name(args) -> str:
    parts = [args.framework, f"D{args.board_dim}", f"N{args.n_tiles}"]
    if args.bias:
        parts.append(f"bias{args.bias:g}")
    if args.framework == "cfm":
        parts.append(f"ode{args.ode_steps}")
    else:
        parts.append(f"T{args.timesteps}")
    parts += [
        f"steps{args.steps}",
        f"d{args.depth}h{args.hidden_dim}",  # includes depth in run name
    ]
    if args.use_rff:
        parts.append(f"rff{args.num_features}h{args.num_harmonics:g}")
    else:
        parts.append("norff")
    parts += [f"lr{args.lr:g}", f"seed{args.seed}"]
    return "_".join(parts)


def build_dataset(args):
    weights = None
    if args.bias > 0:
        idx = np.indices((args.n_tiles,) * args.board_dim)
        active = (idx.sum(axis=0) % 2) == 0
        dist = np.sqrt((idx**2).sum(axis=0))
        weights = active * np.exp(-args.bias * dist)
    return Checkerboard(args.board_dim, args.n_tiles, weights=weights)


def build_scheduler(args):
    return DDPMScheduler(T=args.timesteps)


def build_model(args, scheduler=None):
    if args.framework == "cfm":
        return FlowMLP(
            dim=args.board_dim,
            hidden_dim=args.hidden_dim,
            time_emb_dim=args.time_emb_dim,
            num_features=args.num_features,
            num_harmonics=args.num_harmonics,
            depth=args.depth,
            n_tiles=args.n_tiles,
            use_rff=args.use_rff,
        )

    spatial_embed = None
    alpha_bars = None
    if args.use_rff:
        spatial_embed = HeatGatedFourierFeatures(
            dim=args.board_dim,
            n_tiles=args.n_tiles,
            num_features=args.num_features,
            num_harmonics=args.num_harmonics,
        )
        alpha_bars = scheduler.alpha_bars

    return MLP(
        dim=args.board_dim,
        hidden_dim=args.hidden_dim,
        time_emb_dim=args.time_emb_dim,
        spatial_embed=spatial_embed,
        alpha_bars=alpha_bars,
        depth=args.depth,
    )


def build_framework(args, scheduler=None):
    if args.framework == "cfm":
        return CFM()
    return DDPM(scheduler)


def main():
    args = parse_args()
    name = args.name or default_run_name(args)
    rundir = args.outdir / name
    rundir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = build_dataset(args)
    scheduler = build_scheduler(args) if args.framework == "ddpm" else None
    device = preferred_device()
    if scheduler is not None:
        scheduler.to(device)
    framework = build_framework(args, scheduler)
    model = build_model(args, scheduler).to(device)

    print(f"run: {name}")
    print(
        f"  framework={args.framework} params={sum(p.numel() for p in model.parameters())}"
    )
    print(f"  device={device}")
    if args.framework == "ddpm":
        print(f"  T={args.timesteps}")
        print(
            f"  alpha_bar: first={scheduler.alpha_bars[0]:.5f} "
            f"last={scheduler.alpha_bars[-1]:.2e}"
        )
        if scheduler.alpha_bars[-1] > 1e-3:
            print(
                "  WARNING: forward process leaves signal undestroyed, so sampling "
                "starts from noise the model never saw in training"
            )

    train(
        model,
        framework,
        dataset,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        rng=rng,
    )

    model.eval()
    if args.framework == "cfm":
        generated = (
            from_model_space(
                framework.sample(model, n=args.n_samples, num_steps=args.ode_steps)
            )
            .cpu()
            .numpy()
        )
        eval_desc = f"{args.ode_steps} ODE steps"
    else:
        generated = (
            from_model_space(framework.sample(model, n=args.n_samples)).cpu().numpy()
        )
        eval_desc = f"{args.timesteps} reverse steps"
    ground_truth = dataset.sample(args.n_samples, rng)

    metrics = evaluate_and_plot(
        generated,
        ground_truth,
        args.board_dim,
        args.n_tiles,
        rundir,
        desc=eval_desc,
        label=f"{args.framework.upper()} samples",
    )

    torch.save({"model": model.state_dict(), "args": vars(args)}, rundir / "model.pt")
    (rundir / "metrics.json").write_text(
        json.dumps(
            {
                "name": name,
                "metrics": metrics,
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
    )
    print(f"\nwrote {rundir}/")


if __name__ == "__main__":
    main()
