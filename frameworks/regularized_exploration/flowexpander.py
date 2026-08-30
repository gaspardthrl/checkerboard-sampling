import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from frameworks.cfm import CFM, CFMTrajectory

__all__ = [
    "FEConfig",
    "FEAdjointTargets",
    "FlowExpansionResult",
    "make_entropy_reward_gradient",
    "compute_lean_adjoint",
    "adjoint_matching_loss",
    "adjoint_matching",
    "expand_then_project",
    "flow_expander",
]


@dataclass
class FEConfig:
    """Finite-grid settings for one Algorithm-4 fine-tuning solve."""

    num_steps: int = 40
    t_min: float = 0.05
    N: int = 4
    m: int = 64
    inner_steps: int = 8
    batch_size: int = 2048
    lr: float = 5e-4
    gradient_clip: Optional[float] = 1.0


@dataclass
class FEAdjointTargets:
    """Stop-gradient AM targets on one full memoryless-SDE trajectory batch."""

    trajectory: CFMTrajectory
    adjoint: torch.Tensor  # (S + 1, m, dim)
    base_velocity: torch.Tensor  # (S, m, dim), at trajectory.states[:-1]


@dataclass
class FlowExpansionResult:
    """Final model and models produced by every Flow-Expander iteration."""

    model: nn.Module
    history: list[dict]


def make_entropy_reward_gradient(
    current_model: nn.Module,
    pretrained_model: nn.Module,
    beta: float,
):
    cfm = CFM()

    @torch.no_grad()
    def reward_gradient(x: torch.Tensor, t: float):
        time = torch.full((len(x),), t, device=x.device, dtype=x.dtype)
        current_velocity = current_model(x, time)
        score = cfm.score_from_velocity(current_velocity, x, time)
        if beta:
            pretrained_velocity = pretrained_model(x, time)
            score = score - beta * cfm.score_from_velocity(pretrained_velocity, x, time)
        return -score

    return reward_gradient


def compute_lean_adjoint(
    base_model: nn.Module,
    trajectory: CFMTrajectory,
    cfm: CFM,
    running_gradient: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    gamma: float = 1.0,
    lambda_t: Callable[[float], float] = lambda t: 1.0,
    terminal_gradient: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    times, states, dt = trajectory.times, trajectory.states, trajectory.dt
    adjoint = torch.empty_like(states)
    current_adjoint = (
        terminal_gradient(states[-1])
        if terminal_gradient is not None
        else torch.zeros_like(states[-1])
    )
    adjoint[-1] = current_adjoint

    for i in range(len(times) - 2, -1, -1):
        next_time = float(times[i + 1])
        next_state = states[i + 1].detach().requires_grad_(True)
        with torch.enable_grad():
            base_drift = cfm.memoryless_sde_drift(base_model, next_state, next_time)
        (drift_vjp,) = torch.autograd.grad(
            base_drift, next_state, grad_outputs=current_adjoint
        )
        reward_weight = gamma * lambda_t(next_time)
        running_reward = (
            reward_weight * running_gradient(states[i + 1], next_time)
            if running_gradient is not None and reward_weight != 0
            else 0
        )
        current_adjoint = (current_adjoint + dt * (drift_vjp + running_reward)).detach()
        adjoint[i] = current_adjoint

    input_times = times[:-1]
    m, dim = states.shape[1:]
    model_times = input_times[:, None].expand(-1, m).reshape(-1)
    with torch.no_grad():
        base_velocity = base_model(states[:-1].reshape(-1, dim), model_times)
    targets = FEAdjointTargets(
        trajectory=CFMTrajectory(times.detach(), states.detach(), dt),
        adjoint=adjoint.detach(),
        base_velocity=base_velocity.reshape(len(input_times), m, dim).detach(),
    )
    assert targets.adjoint.shape == targets.trajectory.states.shape
    assert targets.base_velocity.shape == targets.trajectory.states[:-1].shape
    assert torch.isfinite(targets.adjoint).all()
    return targets


def adjoint_matching_loss(
    model: nn.Module,
    targets: FEAdjointTargets,
    step_idx: torch.Tensor,
    trajectory_idx: torch.Tensor,
    cfm: CFM,
):
    state = targets.trajectory.states[step_idx, trajectory_idx]
    time = targets.trajectory.times[step_idx]
    velocity = model(state, time)
    delta_velocity = velocity - targets.base_velocity[step_idx, trajectory_idx]
    noise_std = torch.as_tensor(
        [cfm.memoryless_noise_std(float(t)) for t in time], device=state.device
    ).unsqueeze(-1)

    residual = (
        2 / noise_std * delta_velocity
        - noise_std * targets.adjoint[step_idx, trajectory_idx]
    )
    loss = residual.square().sum(dim=-1).mean()
    assert torch.isfinite(loss)
    return loss


def adjoint_matching(
    base_model: nn.Module,
    cfg: FEConfig,
    running_gradient: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    gamma: float = 1.0,
    lambda_t: Callable[[float], float] = lambda t: 1.0,
    terminal_gradient: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    reference_model = copy.deepcopy(base_model).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    model = copy.deepcopy(reference_model).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    diagnostics = []

    cfm = CFM()

    for _ in range(cfg.N):
        trajectory = cfm.sample(
            model,
            n=cfg.m,
            num_steps=cfg.num_steps,
            full=True,
            memoryless_sde=True,
            t_min=cfg.t_min,
        )  # Line 3
        targets = compute_lean_adjoint(
            reference_model,
            trajectory,
            cfm,
            running_gradient,
            gamma,
            lambda_t,
            terminal_gradient,
        )  # Line 4
        metrics = {"adjoint_norm": targets.adjoint[:-1].norm(dim=-1).mean().item()}

        # AM SGD samples the explicit (time-step, trajectory) pairs.
        # Similar to SMEME
        pair_count = cfg.num_steps * cfg.m
        for _ in range(cfg.inner_steps):
            pair = torch.randint(
                pair_count,
                (min(cfg.batch_size, pair_count),),
                device=targets.trajectory.states.device,
            )
            step_idx, trajectory_idx = pair // cfg.m, pair % cfg.m
            loss = adjoint_matching_loss(model, targets, step_idx, trajectory_idx, cfm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            metrics["loss"] = loss.item()
        diagnostics.append(metrics)

    return model.eval(), diagnostics


def expand_then_project(
    current_model: nn.Module,
    cfg: FEConfig,
    reward_gradient: Callable[[torch.Tensor, float], torch.Tensor],
    gamma: float,
    lambda_t: Callable[[float], float],
    eta: float,
    verifier: Optional[dict] = None,
):
    expanded, _ = adjoint_matching(
        current_model,
        cfg,
        running_gradient=reward_gradient,
        gamma=gamma,
        lambda_t=lambda_t,
    )

    # If eta is 0 we are doing NSE which discards projection steps.
    if eta == 0 or verifier is None:
        return expanded

    projected, _ = adjoint_matching(
        expanded,
        cfg,
        terminal_gradient=lambda x: eta * verifier["grad_log_v"](x),
    )
    return projected


def flow_expander(
    pretrained: nn.Module,
    alphas: Sequence[float],
    gammas: Sequence[float],
    etas: Sequence[float],
    verifier: Optional[dict],
    cfg: Optional[FEConfig] = None,
    lambda_t: Callable[[float], float] = lambda t: float(t < 0.95),
    return_history: bool = False,
):
    cfg = cfg or FEConfig()
    reference = copy.deepcopy(pretrained).eval()
    current = copy.deepcopy(pretrained).eval()
    history = []
    for alpha, gamma, eta in zip(alphas, gammas, etas):
        # Appendix H: beta = alpha / (1 + alpha) and
        # gamma_tilde = (1 + alpha) gamma. Alpha is constant over time.
        beta = alpha / (1 + alpha)
        gamma_tilde = (1 + alpha) * gamma
        reward_gradient = make_entropy_reward_gradient(current, reference, beta)
        current = expand_then_project(
            current, cfg, reward_gradient, gamma_tilde, lambda_t, eta, verifier
        )
        history.append(
            {
                "model": current,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "gamma_tilde": gamma_tilde,
                "eta": eta,
            }
        )
    result = FlowExpansionResult(model=current, history=history)
    return result if return_history else result.model
