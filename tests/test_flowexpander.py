"""Flow Expander correctness, pinned against closed-form answers.

Read this before touching signs in `flowexpander.py`.

The FE paper cannot be used as the reference for the sign conventions, because
its Eq. 52 is internally inconsistent: the running term enters the lean-adjoint
recursion with a MINUS while the terminal condition is stated with a PLUS
(+gamma*lambda*grad f). Under one convention those cannot both hold. Eq. 52 also
carries an explicit gamma*lambda while Eq. 59/60 already fold gamma-tilde and
lambda_t into grad f, so following both literally squares the step size.

`test_terminal_reward_tilts_by_exactly_the_h_transform` resolves it empirically
and exactly, so the implementation is anchored to arithmetic rather than to the
paper's typography.
"""

import math

import pytest
import torch

from frameworks.cfm import CFM
from frameworks.regularized_exploration.flowexpander import (
    FEConfig,
    adjoint_matching,
    expand_then_project,
    flow_expander,
    make_entropy_reward_gradient,
)

from .analytic import ControlledFlow, GaussianFlow, ode_sample

DIM = 2


def memoryless_trajectory(model, n, cfg):
    return CFM().sample(
        model,
        n=n,
        num_steps=cfg.num_steps,
        full=True,
        memoryless_sde=True,
        t_min=cfg.t_min,
    )


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Interpolant algebra                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.5, 0.4), (-0.3, 0.25)])
@pytest.mark.parametrize("t", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_score_from_velocity_matches_the_analytic_score(mean, std, t):
    """s_t(x) = (t u - x)/(1-t), paper Eq. 19 at kappa=1-t, omega=t."""
    flow = GaussianFlow(DIM, mean, std)
    x = torch.randn(400, DIM) * 1.5
    tb = torch.full((400,), t)
    got = CFM().score_from_velocity(flow(x, tb), x, tb)
    assert torch.allclose(got, flow.score(x, tb), atol=1e-4)


def test_score_at_t_zero_is_the_standard_normal_score():
    flow = GaussianFlow(DIM, 0.7, 0.3)
    x = torch.randn(200, DIM)
    tb = torch.zeros(200)
    assert torch.allclose(CFM().score_from_velocity(flow(x, tb), x, tb), -x, atol=1e-5)


@pytest.mark.parametrize("t", [0.05, 0.3, 0.7, 0.99])
def test_memoryless_sigma_matches_the_interpolant_formula(t):
    """sigma^2 = 2 kappa_t (omega_dot/omega kappa_t - kappa_dot) = 2(1-t)/t."""
    assert CFM().memoryless_noise_std(t) == pytest.approx(math.sqrt(2 * (1 - t) / t))


def test_sde_drift_is_velocity_plus_half_sigma_squared_score():
    """b = u + (sigma^2/2) s, which collapses to 2u - x/t. Verify the identity."""
    flow = GaussianFlow(DIM, 0.2, 0.6)
    x = torch.randn(300, DIM)
    for t in (0.1, 0.4, 0.9):
        tb = torch.full((300,), t)
        u = flow(x, tb)
        cfm = CFM()
        expected = u + 0.5 * cfm.memoryless_noise_std(t) ** 2 * cfm.score_from_velocity(u, x, tb)
        assert torch.allclose(cfm.memoryless_sde_drift(flow, x, t), expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# The memoryless SDE must reproduce the ODE marginals                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.5, 0.4)])
def test_memoryless_sde_reproduces_the_flow_marginals(mean, std):
    """If this fails, every reward and adjoint downstream is evaluated on the
    wrong states and no amount of sign-fixing will help."""
    flow = GaussianFlow(DIM, mean, std)
    cfg = FEConfig(num_steps=40, t_min=0.05)
    terminal = memoryless_trajectory(flow, 20_000, cfg).states[-1]
    assert terminal.mean(0).tolist() == pytest.approx([mean] * DIM, abs=0.03)
    assert terminal.std(0).tolist() == pytest.approx([std] * DIM, rel=0.04)


def test_cfm_full_sde_history_has_the_expected_grid():
    cfg = FEConfig(num_steps=8, t_min=0.05)
    trajectory = memoryless_trajectory(GaussianFlow(DIM), 16, cfg)
    assert trajectory.states.shape == (cfg.num_steps + 1, 16, DIM)
    assert trajectory.times[0].item() == pytest.approx(cfg.t_min)
    assert trajectory.times[-1].item() == pytest.approx(1.0)
    assert trajectory.dt == pytest.approx((1.0 - cfg.t_min) / cfg.num_steps)


def test_cfm_full_ode_history_ends_at_the_ordinary_sample():
    flow = GaussianFlow(DIM, 0.2, 0.6)
    cfm = CFM()
    torch.manual_seed(4)
    trajectory = cfm.sample(flow, n=16, num_steps=8, full=True)
    torch.manual_seed(4)
    endpoint = cfm.sample(flow, n=16, num_steps=8)
    assert trajectory.states.shape == (9, 16, DIM)
    assert torch.allclose(trajectory.states[-1], endpoint)


# --------------------------------------------------------------------------- #
# THE sign test                                                                #
# --------------------------------------------------------------------------- #


def test_terminal_reward_tilts_by_exactly_the_h_transform():
    """Projection-step sign check, with an exact closed-form target.

    Base terminal law is N(0, I). Fine-tuning against the linear terminal reward
    r(x) = <w, x> must produce the h-transform tilt

        p ∝ N(0,I) e^{<w,x>} = N(w, I),

    i.e. the terminal mean moves to exactly +w and the covariance is unchanged.
    Integrating the controlled mean ODE gives the tilt factor
    int_0^1 (1/V^2 - 1/V) dt = 1 exactly, with V(t) = (1-t)^2 + t^2; the ODE
    warm-up on [0, t_min] costs 0.3% of it at t_min = 0.05.

    A flipped sign anywhere in the chain sends the mean to -w, and a lost factor
    of 2 in the residual sends it to w/2 or 2w. This is the test that says the
    paper's Eq. 52 minus is the typo, not the implementation.
    """
    w = torch.tensor([0.8, -0.5])
    base = ControlledFlow(GaussianFlow(DIM, 0.0, 1.0), DIM)
    cfg = FEConfig(num_steps=40, t_min=0.05, N=40, m=512, inner_steps=25, lr=3e-3)

    tuned, _ = adjoint_matching(
        base, cfg,
        terminal_gradient=lambda x: w.to(x).expand_as(x),
    )

    for name, x in (("SDE", memoryless_trajectory(tuned, 20_000, cfg).states[-1]),
                    ("ODE", ode_sample(tuned, 20_000, DIM, 200))):
        assert x.mean(0).tolist() == pytest.approx(w.tolist(), abs=0.08), name
        assert x.std(0).tolist() == pytest.approx([1.0, 1.0], rel=0.06), name


def test_flipping_the_terminal_gradient_flips_the_shift():
    """Guards the direction itself, cheaply: -w must move the mean the other way."""
    w = torch.tensor([1.0, 0.0])
    cfg = FEConfig(num_steps=40, t_min=0.05, N=20, m=512, inner_steps=20, lr=3e-3)
    means = []
    for sign in (+1.0, -1.0):
        torch.manual_seed(0)
        base = ControlledFlow(GaussianFlow(DIM, 0.0, 1.0), DIM)
        tuned, _ = adjoint_matching(
            base, cfg,
            terminal_gradient=lambda x, s=sign: (s * w).to(x).expand_as(x),
        )
        means.append(ode_sample(tuned, 8000, DIM, 100).mean(0)[0].item())
    assert means[0] > 0.5 and means[1] < -0.5


# --------------------------------------------------------------------------- #
# Expansion step: the entropy reward must raise entropy                        #
# --------------------------------------------------------------------------- #


def test_expansion_increases_spread_and_gamma_zero_is_a_no_op():
    """Running-reward sign check.

    grad delta H = -s_t, so the reward pushes mass down the score, i.e. outward.
    A flipped sign would *contract* the distribution -- entropy minimisation --
    which is exactly the failure mode the S-MEME module's docstring warns about.
    The base is isotropic and centred, so the mean must not move.
    """
    base_std = 0.5
    variances = {}
    for gamma in (0.0, 0.5, 1.0):
        torch.manual_seed(0)
        base = ControlledFlow(GaussianFlow(DIM, 0.0, base_std), DIM)
        cfg = FEConfig(
            num_steps=40, t_min=0.05, N=25, m=512, inner_steps=20, lr=3e-3,
        )
        reward_gradient = make_entropy_reward_gradient(
            base, base, 0.0
        )
        out = expand_then_project(
            current_model=base,
            cfg=cfg,
            reward_gradient=reward_gradient,
            gamma=gamma,
            lambda_t=lambda t: 1.2 if t < 0.95 else 0.0,
            eta=0.0,
            verifier=None,
        )
        x = ode_sample(out, 12_000, DIM, 150)
        variances[gamma] = x.var(0).mean().item()
        assert x.mean(0).abs().max() < 0.12, "expansion must stay centred"

    assert variances[0.0] == pytest.approx(base_std**2, rel=0.08), "gamma=0 is a no-op"
    assert variances[0.5] > 1.15 * variances[0.0]
    assert variances[1.0] > 1.6 * variances[0.5]


def test_eta_zero_skips_the_projection_step():
    """NSE is FE with the projection removed (Alg. 3), so the verifier must be
    ignored entirely when eta = 0 -- not merely down-weighted."""
    base = ControlledFlow(GaussianFlow(DIM, 0.0, 0.5), DIM)
    cfg = FEConfig(num_steps=10, t_min=0.05, N=1, m=64, inner_steps=1)

    def exploding_verifier(_):
        raise AssertionError("verifier must not be consulted when eta = 0")

    reward_gradient = make_entropy_reward_gradient(
        base, base, 0.0
    )
    expand_then_project(
        current_model=base,
        cfg=cfg,
        reward_gradient=reward_gradient,
        gamma=0.1,
        lambda_t=lambda t: float(t < 0.95),
        eta=0.0,
        verifier={"grad_log_v": exploding_verifier, "log_v": exploding_verifier,
                  "hard": exploding_verifier},
    )


def test_flow_expander_records_the_appendix_h_reparameterization(monkeypatch):
    """Public alpha/gamma inputs become beta and gamma_tilde exactly once."""
    calls = []

    def record_expand(current, cfg, reward_gradient, gamma, lambda_t, eta, verifier):
        calls.append((gamma, eta))
        return current

    monkeypatch.setattr(
        "frameworks.regularized_exploration.flowexpander.expand_then_project",
        record_expand,
    )
    base = GaussianFlow(DIM, 0.0, 1.0)
    result = flow_expander(
        base,
        alphas=[3.0],
        gammas=[0.2],
        etas=[0.7],
        verifier=None,
        return_history=True,
    )
    assert calls == [(0.8, 0.7)]
    assert result.history[0]["beta"] == pytest.approx(0.75)
    assert result.history[0]["gamma_tilde"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# THE expansion-step test                                                      #
# --------------------------------------------------------------------------- #


def test_lean_adjoint_converges_to_the_exact_h_transform():
    """Quantitative check of the RUNNING (expansion) reward path.

    `test_expansion_increases_spread...` only checks that entropy goes up, and a
    distribution that collapses onto a long thin line would pass that test. This
    test pins the expansion against an exact answer instead.

    For a 1-D Gaussian base the whole control problem is linear-quadratic, so
    the h-transform is available in closed form via a Riccati ODE and the
    optimal `a` is exactly alpha(t)*x. The lean adjoint is an approximation
    (states are treated as constants), so it is only exact in the limit of a
    fine SDE grid -- hence the refinement check rather than a single tolerance.
    """
    import numpy as np

    from tests.analytic import riccati_alpha

    STD, GAMMA, LAM = 0.5, 0.3, 1.2
    lam_fn = lambda t: 0.0 if t > 0.95 else LAM  # noqa: E731
    ts_r, alpha = riccati_alpha(GAMMA, lam_fn, std=STD)
    alpha_at = lambda t: float(np.interp(t, ts_r[::-1], alpha[::-1]))  # noqa: E731

    base = GaussianFlow(1, 0.0, STD)

    def running_grad(x, t):
        if t > 0.95:
            return torch.zeros_like(x)
        return GAMMA * LAM * x / ((1 - t) ** 2 + t**2 * STD**2)

    def ratio_at(num_steps, target_t):
        torch.manual_seed(0)
        cfg = FEConfig(num_steps=num_steps, t_min=0.05)
        trajectory = memoryless_trajectory(base, 6000, cfg)
        ts, xs, h = trajectory.times, trajectory.states, trajectory.dt
        a = torch.zeros_like(xs[num_steps])
        store = {}
        for i in range(num_steps - 1, -1, -1):
            x = xs[i + 1].detach().requires_grad_(True)
            with torch.enable_grad():
                b = CFM().memoryless_sde_drift(base, x, float(ts[i + 1]))
            (vjp,) = torch.autograd.grad(b, x, grad_outputs=a)
            a = (a + h * (vjp + running_grad(xs[i + 1], ts[i + 1]))).detach()
            store[i] = a
        i = min(range(num_steps), key=lambda k: abs(ts[k] - target_t))
        x = xs[i].squeeze(-1).numpy()
        av = store[i].squeeze(-1).numpy()
        slope = float((x * av).sum() / (x * x).sum())
        return slope / alpha_at(ts[i])

    coarse = ratio_at(40, 0.76)
    fine = ratio_at(300, 0.76)
    assert 0.80 < coarse < 1.0, f"coarse grid ratio {coarse}"
    assert abs(fine - 1.0) < 0.06, (
        f"lean adjoint does not converge to the exact h-transform: ratio {fine}"
    )
    assert fine > coarse, "refining the grid must move the adjoint toward exact"
