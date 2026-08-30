"""Cheap algebra and shape checks for the S-MEME Algorithm-2 implementation."""

import pytest
import torch
import torch.nn as nn

from frameworks.regularized_exploration.smeme import (
    AMTargets,
    adjoint_matching_loss,
    compute_lean_adjoint,
)
from frameworks.ddpm import DDPM
from frameworks.scheduler import DDPMScheduler


class TinyEpsilon(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x, t):
        return self.linear(x)


@pytest.mark.parametrize("respaced", [False, True])
def test_reversed_schedule_mean_matches_ddpm_step(respaced):
    """Algorithm-2 reverse mean must be exactly DDPMScheduler.step's mean."""
    scheduler = DDPMScheduler(T=12, beta_start=1e-4, beta_end=2e-2)
    if respaced:
        scheduler.respace(torch.tensor([0, 2, 5, 8, 11]))
    x, eps = torch.randn(7, 2), torch.randn(7, 2)

    for j in range(scheduler.T):
        standard_index = scheduler.T - 1 - j
        got = scheduler.step_mean(eps, standard_index, x)
        want = (
            x - scheduler.betas[standard_index]
            * eps / torch.sqrt(1 - scheduler.alpha_bars[standard_index])
        ) / torch.sqrt(scheduler.alphas[standard_index])
        # The deterministic clean transition combines float32 cumulative-alpha
        # values in a different order, so allow only round-off-level error.
        assert torch.allclose(got, want, atol=2e-6, rtol=1e-5), (respaced, j)


@pytest.mark.parametrize("respaced", [False, True])
def test_full_ddpm_trajectory_is_the_algorithm_two_trajectory(respaced):
    """``DDPM.sample(full=True)`` retains the exact noise-to-clean chain AM needs."""
    scheduler = DDPMScheduler(T=12, beta_start=1e-4, beta_end=2e-2)
    if respaced:
        scheduler.respace(torch.tensor([0, 2, 5, 8, 11]))
    model = TinyEpsilon(dim=2)
    sampler = DDPM(scheduler)

    torch.manual_seed(7)
    trajectory = sampler.sample(model, n=5, full=True)
    torch.manual_seed(7)
    x = torch.randn(5, 2)
    expected = [x]
    labels = getattr(scheduler, "timesteps", torch.arange(scheduler.T))
    for chain_index in range(scheduler.T - 1, -1, -1):
        timestep = torch.full((5,), int(labels[chain_index]), dtype=torch.long)
        x = scheduler.step(model(x, timestep), chain_index, x)
        expected.append(x)

    assert torch.equal(trajectory, torch.stack(expected))


def test_algorithm_two_targets_and_loss_have_expected_shapes():
    torch.manual_seed(0)
    model = TinyEpsilon(dim=2)
    scheduler = DDPMScheduler(T=8)
    trajectory = DDPM(scheduler).sample(model, n=5, full=True)
    targets = compute_lean_adjoint(
        model, trajectory, grad_f=lambda x: torch.ones_like(x), scheduler=scheduler
    )
    assert trajectory.shape == (scheduler.T + 1, 5, 2)
    assert targets.adjoint.shape == trajectory.shape
    assert targets.base_eps.shape == (scheduler.T, 5, 2)

    timestep_idx = torch.tensor([0, 1, scheduler.T - 2])
    trajectory_idx = torch.tensor([0, 2, 4])
    loss = adjoint_matching_loss(model, targets, timestep_idx, trajectory_idx, scheduler)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_adjoint_matching_loss_uses_the_paper_timestep_coefficients():
    """The squared loss contains the paper's outer c_t timestep weighting.

    The project intentionally retains the entropy-maximising plus sign, so the
    expected residual below is ``q_eps * Delta + q_adjoint * a``.  This test is
    about the magnitude coefficients only.
    """
    torch.manual_seed(3)
    scheduler = DDPMScheduler(T=8)
    model = TinyEpsilon(dim=2)
    m, dim = 4, 2
    targets = AMTargets(
        trajectory=torch.randn(scheduler.T + 1, m, dim),
        adjoint=torch.randn(scheduler.T + 1, m, dim),
        base_eps=torch.randn(scheduler.T, m, dim),
    )
    j = torch.tensor([0, 2, scheduler.T - 2])
    b = torch.tensor([1, 3, 0])

    got = adjoint_matching_loss(model, targets, j, b, scheduler)

    chain_index, model_timestep = scheduler.reverse_indices(j)
    alpha_bar_t = scheduler.alpha_bars[chain_index]
    previous = (chain_index - 1).clamp_min(0)
    alpha_bar_next = torch.where(
        chain_index > 0, scheduler.alpha_bars[previous], torch.ones_like(alpha_bar_t)
    )
    alpha_ratio = alpha_bar_next / alpha_bar_t
    c = 1.0 - 1.0 / alpha_ratio
    paper_eps_weight = c * (alpha_ratio / (1.0 - alpha_bar_next)).sqrt()
    paper_adjoint_weight = c * ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)).sqrt()
    state = targets.trajectory[j, b]
    delta_eps = model(state, model_timestep) - targets.base_eps[j, b]
    expected_residual = (
        paper_eps_weight[:, None] * delta_eps
        + paper_adjoint_weight[:, None] * targets.adjoint[j, b]
    )
    expected = expected_residual.square().sum(dim=-1).mean()

    torch.testing.assert_close(got, expected)
