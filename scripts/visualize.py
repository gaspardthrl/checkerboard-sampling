"""Load a trained run and re-plot / re-evaluate its samples.

Sampling uses the same T steps and the same time grid the model was trained on,
so this reproduces the run's evaluation without retraining.

Examples:
    uv run python scripts/visualize.py outputs/runs/<name>
    uv run python scripts/visualize.py outputs/runs/<name> --n-samples 50000 --n-slices 8
"""

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from frameworks.ddpm import DDIM, DDPM
from models.mlp import MLP, HeatGatedFourierFeatures
from scripts.train import build_dataset, build_scheduler
from utils.normalize import from_model_space
from utils.device import preferred_device
from utils.report import evaluate_and_plot


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", type=Path, help="run directory or path to model.pt")
    p.add_argument("--n-samples", type=int, default=20000)
    p.add_argument(
        "--n-slices",
        type=int,
        default=None,
        help="z bands to split into (default: one per tile row)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sampler",
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="ddim reuses the same trained model deterministically, "
        "no retraining needed",
    )
    return p.parse_args()


def load_run(path: Path):
    ckpt_path = path / "model.pt" if path.is_dir() else path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = Namespace(**ckpt["args"])
    state = ckpt["model"]

    # infer from the checkpoint itself (self-describing) rather than trusting
    # saved args to still match this MLP's current parameter names
    use_rff = "spatial_embed.freqs" in state
    dim = getattr(targs, "board_dim", 3)
    scheduler = build_scheduler(targs)

    spatial_embed = None
    alpha_bars = None
    if use_rff:
        # num_features is read back from the checkpoint's own buffer shape;
        # num_harmonics doesn't matter here since freqs/phases get overwritten
        # by load_state_dict right below regardless of what we construct with
        spatial_embed = HeatGatedFourierFeatures(
            dim=dim,
            n_tiles=targs.n_tiles,
            num_features=state["spatial_embed.freqs"].shape[0],
        )
        alpha_bars = scheduler.alpha_bars

    # freqs/phases are buffers, so load_state_dict restores the trained projection
    model = MLP(
        dim=dim,
        hidden_dim=targs.hidden_dim,
        time_emb_dim=targs.time_emb_dim,
        spatial_embed=spatial_embed,
        alpha_bars=alpha_bars,
        depth=getattr(targs, "depth", 4),
    )
    model.load_state_dict(state)
    model.eval()
    return targs, build_dataset(targs), model, scheduler, ckpt_path.parent


def main():
    args = parse_args()
    targs, dataset, model, scheduler, rundir = load_run(args.run)
    device = preferred_device()
    model.to(device)
    scheduler.to(device)
    dim = getattr(targs, "board_dim", 3)
    print(f"loaded {rundir}")
    print(f"  device={device}")
    print(
        f"  trained: dim={dim} N={targs.n_tiles} bias={targs.bias} "
        f"T={targs.timesteps} steps={targs.steps}"
    )

    sampler = DDIM(scheduler) if args.sampler == "ddim" else DDPM(scheduler)
    torch.manual_seed(args.seed)

    generated = from_model_space(sampler.sample(model, n=args.n_samples)).numpy()
    ground_truth = dataset.sample(args.n_samples, np.random.default_rng(args.seed))

    evaluate_and_plot(
        generated,
        ground_truth,
        dim,
        targs.n_tiles,
        rundir,
        desc=f"{args.sampler}, {targs.timesteps} reverse steps",
        label=f"{args.sampler.upper()} samples",
        suffix=f"_{args.sampler}",
        n_slices=args.n_slices,
    )
    print(f"wrote plots to {rundir}/")


if __name__ == "__main__":
    main()
