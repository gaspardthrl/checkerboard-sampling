import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from frameworks.ddpm import DDPM
from frameworks.scheduler import DDPMScheduler

__all__ = [
    "AMTargets",
    "SMEMEResult",
    "make_entropy_reward_gradient",
    "compute_lean_adjoint",
    "adjoint_matching_loss",
    "adjoint_matching",
    "smeme",
]


@dataclass
class AMTargets:
    """Stop-gradient regression targets from one Algorithm-2 trajectory batch."""

    trajectory: torch.Tensor  # (T + 1, m, dim), DDPM.sample(..., full=True)
    adjoint: torch.Tensor  # (T + 1, m, dim)
    base_eps: torch.Tensor  # (T, m, dim), epsilon_ref(X_j, t_j)


@dataclass
class SMEMEResult:
    """Algorithm-1 output plus runner-only diagnostics."""

    model: nn.Module
    reward_timestep: int
    reward_noise_std: float
    history: list[dict]

    def __iter__(self):
        """Keep ``model, history = smeme(..., return_history=True)`` working."""
        yield self.model
        yield self.history


def make_entropy_reward_gradient(
    base_model: nn.Module,
    alpha: float,
    reward_timestep: int,
    reward_noise_std: float,
    clip_norm: Optional[float] = None,
):
    @torch.no_grad()
    def grad_f(x: torch.Tensor):
        timestep = torch.full(
            (len(x),), reward_timestep, dtype=torch.long, device=x.device
        )
        eps = base_model(x, timestep)
        score = -eps / reward_noise_std
        gradient = -score / alpha
        if clip_norm is not None:
            norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            gradient = gradient * (norm.clamp(max=clip_norm) / norm)
        return gradient

    return grad_f


def _reference_drift(
    model: nn.Module, x: torch.Tensor, j: int, scheduler: DDPMScheduler
):
    chain_index, model_timestep = scheduler.reverse_indices(j)
    timestep = torch.full((len(x),), model_timestep, dtype=torch.long, device=x.device)
    eps = model(x, timestep)
    mean = scheduler.step_mean(eps, chain_index, x)
    return mean - x, eps


def _am_coefficients(scheduler: DDPMScheduler, j: torch.Tensor, like: torch.Tensor):
    """Appendix-D residual coefficients in the paper's reverse trajectory order."""
    chain_index, _ = scheduler.reverse_indices(j)
    chain_index = chain_index.to(scheduler.alpha_bars.device)

    alpha_bar_t = scheduler.alpha_bars[chain_index].to(like)  # \bar{\alpha}_t

    previous_index = (chain_index - 1).clamp_min(0)
    previous_alpha_bar = scheduler.alpha_bars[previous_index].to(like)
    alpha_bar_next = torch.where(
        (chain_index > 0).to(like.device),
        previous_alpha_bar,
        torch.ones_like(alpha_bar_t),
    )
    alpha_ratio = (
        alpha_bar_next / alpha_bar_t
    )  # \frac{\bar{\alpha}_{t+1}}{\bar{\alpha}_t}
    c = 1.0 - 1.0 / alpha_ratio  # c := (1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t+1}})
    posterior_sigma = (
        (c * (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)).clamp_min(0).sqrt()
    )
    control_weight = (alpha_ratio * c / (1.0 - alpha_bar_next).clamp_min(1e-20)).sqrt()
    return control_weight, posterior_sigma, c.sqrt()


def compute_lean_adjoint(
    base_model: nn.Module,
    trajectory: torch.Tensor,
    grad_f: Callable[[torch.Tensor], torch.Tensor],
    scheduler: DDPMScheduler,
):
    T, m = scheduler.T, trajectory.shape[1]
    adjoint = torch.empty_like(trajectory)

    # \epsilon^{pre} is used to solve the lean adjoint ODE and (\epsilon^{pre}(X_t, t) ; \bar{a}_t) are used to compute the loss
    # Thus we need to keep the pairs and not recompute \epsilon^{pre}(X_t, t)
    base_eps = torch.empty(T, m, trajectory.shape[-1], device=trajectory.device)

    # Lean Adjoint ODE initial condition (Use reward gradient ...)
    current_adjoint = grad_f(trajectory[-1])
    adjoint[-1] = current_adjoint

    # Solve the lean adjoint ODE from t = T to 0.
    for j in range(T - 1, -1, -1):
        Xj = trajectory[j].detach().requires_grad_(True)

        # eps:= \epsilon^{\rm pre}(X_t,t)
        with torch.enable_grad():
            drift, eps = _reference_drift(base_model, Xj, j, scheduler)

        # Compute gradient of drift w.r.t. state and computes vector Jacobian product with current_adjoint
        (vjp,) = torch.autograd.grad(drift, Xj, grad_outputs=current_adjoint)

        base_eps[j] = eps.detach()
        current_adjoint = (
            current_adjoint + vjp
        ).detach()  # \bar{a}_t = \bar{a}_{t+1} + \bar{a}_{t+1}^T \nabla_{X_t} drift (vjp)
        adjoint[j] = current_adjoint

    targets = AMTargets(trajectory.detach(), adjoint.detach(), base_eps)
    return targets


def adjoint_matching_loss(
    model: nn.Module,
    targets: AMTargets,
    timestep_idx: torch.Tensor,
    trajectory_idx: torch.Tensor,
    scheduler: DDPMScheduler,
):
    """Appendix-D AM regression on fixed ``(j, trajectory)`` pairs."""
    state = targets.trajectory[timestep_idx, trajectory_idx]  # X_t
    _, model_timestep = scheduler.reverse_indices(timestep_idx)
    model_timestep = model_timestep.to(state.device)

    eps = model(state, model_timestep)  # \eps^{finetuned}(X_t, t)
    delta_eps = eps - targets.base_eps[timestep_idx, trajectory_idx]

    control_weight, posterior_sigma, sqrt_c = _am_coefficients(
        scheduler, timestep_idx, state
    )

    # Paper magnitude is sqrt(c_j)[w_j delta_eps - sigma_j a_j]. Preserve the
    # project's entropy-maximising + sign while applying the paper's c_j weight.
    residual = sqrt_c.unsqueeze(-1) * (
        control_weight.unsqueeze(-1) * delta_eps
        + posterior_sigma.unsqueeze(-1) * targets.adjoint[timestep_idx, trajectory_idx]
    )
    loss = residual.square().sum(dim=-1).mean()
    assert torch.isfinite(loss)
    return loss


def adjoint_matching(
    base_model: nn.Module,
    grad_f: Callable[[torch.Tensor], torch.Tensor],
    *,
    scheduler: DDPMScheduler,
    N: int,
    m: int,
    inner_steps: int = 2,
    batch_size: int = 2048,
    lr: float = 4e-4,
    gradient_clip: Optional[float] = 1.0,
):
    # \epsilon^{pre} (frozen)
    reference_model = copy.deepcopy(base_model).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    # Line 1. \epsilon^{finetuned} (trainable)
    model = copy.deepcopy(reference_model).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = DDPM(scheduler)
    diagnostics = []

    # The final deterministic clean transition has 1-bar_alpha_{t+1}=0, making
    # the AM control coefficient singular. Sample only nonterminal pairs (j, b).
    pair_count = (scheduler.T - 1) * m

    # Line 2.
    for _ in range(N):
        # Line 3.a (Sample m trajectories ...)
        trajectory = sampler.sample(model, n=m, full=True)

        # Line 3.b (For each trajectory solve the lean adjoint ODE ...) & Computes the initial condition \bar{a}_T
        targets = compute_lean_adjoint(reference_model, trajectory, grad_f, scheduler)

        metrics = {
            "grad_f_norm": targets.adjoint[-1].norm(dim=-1).mean().item(),
            "adjoint_norm": targets.adjoint[:-1].norm(dim=-1).mean().item(),
        }

        # SGD
        for _ in range(inner_steps):
            pair = torch.randint(
                0, pair_count, (min(batch_size, pair_count),), device=trajectory.device
            )
            j, b = pair // m, pair % m
            loss = adjoint_matching_loss(model, targets, j, b, scheduler)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            metrics["loss"] = loss.item()
        diagnostics.append(metrics)

    return model.eval(), diagnostics


def smeme(
    K: int,
    eps_pre: nn.Module,
    alphas: Sequence[float],
    scheduler: DDPMScheduler,
    N: int = 30,
    m: int = 20,
    inner_steps: int = 2,
    batch_size: int = 2048,
    lr: float = 4e-4,
    reward_clip: Optional[float] = 10.0,
    reward_noise_std: Optional[float] = None,
    return_history: bool = False,
):
    # Below is not in the paper but is a safety margin.
    # Paper's reward is computed at t = 0 (i.e. zero noise). Score can be ill-defined at t=0.
    # We compute reward at reward_timestep i.e. timestep which noise is closest to reward_noise_std
    noise_levels = torch.sqrt(1.0 - scheduler.alpha_bars)
    score_index = (
        0
        if reward_noise_std is None
        else int(torch.argmin((noise_levels - reward_noise_std).abs()).item())
    )
    reward_timestep = int(scheduler.timesteps[score_index])
    reward_std = float(noise_levels[score_index])  # Actual noise at reward_timestep

    model = copy.deepcopy(eps_pre)
    history = []
    for k in range(K):
        # Algorithm 1: the current model is pi_k and defines grad f_k.
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        grad_f = make_entropy_reward_gradient(
            model, float(alphas[k]), reward_timestep, reward_std, reward_clip
        )

        # LinearFinetuningSolver
        model, diagnostics = adjoint_matching(
            model,
            grad_f,
            scheduler=scheduler,
            N=N,
            m=m,
            inner_steps=inner_steps,
            batch_size=batch_size,
            lr=lr,
        )

        history.append(
            {"k": k, "alpha": float(alphas[k]), "model": model, "log": diagnostics}
        )

    result = SMEMEResult(model, reward_timestep, reward_std, history)
    return result if return_history else result.model
