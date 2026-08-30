"""Gaussian flows with closed-form velocity, score, and h-transform.

Everything in flow expansion is a chain of sign conventions -- score from
velocity, adjoint recursion, regression residual -- and the FE paper's Eq. 52 is
self-inconsistent about it (the running term carries a minus while the terminal
condition carries a plus; those cannot both hold). So the conventions have to be
pinned by something that has an exact answer, not by reading the paper harder.

For the linear interpolant X_t = (1-t) X_0 + t X_1 with

    X_0 ~ N(0, I)   and   X_1 ~ N(mean, std^2 I)   independent,

X_t is Gaussian with mean b*mean and variance V(t) = a^2 + b^2 std^2 where
a = 1-t, b = t, and every quantity below is available in closed form.
"""

import math

import torch
import torch.nn as nn


class GaussianFlow(nn.Module):
    """Exact marginal velocity field u(x,t) = E[X_1 - X_0 | X_t = x].

    Deliberately parameter-free: it is the ground-truth base field, and having
    no parameters means a test that accidentally trains it will fail loudly
    rather than quietly fitting the reference.
    """

    def __init__(self, dim: int, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("mean", torch.full((dim,), float(mean)))
        self.std = float(std)

    def var(self, t):
        a, b = 1.0 - t, t
        return a**2 + b**2 * self.std**2

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tc = t.view(-1, 1).to(x)
        a, b = 1.0 - tc, tc
        V = a**2 + b**2 * self.std**2
        m = self.mean.to(x)
        return m + ((b * self.std**2 - a) / V) * (x - b * m)

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """grad log p_t(x), independently derived (not via the velocity)."""
        tc = t.view(-1, 1).to(x)
        return -(x - tc * self.mean.to(x)) / self.var(tc)

    def terminal_entropy(self) -> float:
        """H(N(mean, std^2 I)) in nats."""
        return 0.5 * self.dim * math.log(2 * math.pi * math.e * self.std**2)


class ControlledFlow(nn.Module):
    """Frozen analytic base plus a trainable additive correction, init to zero.

    This is what lets an exact base field be handed to `finetuning_solver`,
    which deep-copies its input and optimises the copy: at init the copy is
    numerically identical to the base, so the solver starts from the right place.
    """

    def __init__(self, base: nn.Module, dim: int, hidden: int = 64):
        super().__init__()
        self.base = base
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tc = t.view(-1, 1).to(x).expand(x.shape[0], 1)
        return self.base(x, t) + self.net(torch.cat([x, tc], dim=-1))


@torch.no_grad()
def ode_sample(model: nn.Module, n: int, dim: int, num_steps: int = 200):
    """Midpoint integration of dx/dt = u(x,t) from t=0 to t=1."""
    x = torch.randn(n, dim)
    h = 1.0 / num_steps
    for i in range(num_steps):
        t0 = torch.full((n,), i * h)
        tm = torch.full((n,), (i + 0.5) * h)
        x = x + h * model(x + 0.5 * h * model(x, t0), tm)
    return x


def riccati_alpha(gamma, lam_fn, std=0.5, t_min=0.05, n=100_000):
    """Exact h-transform for the FE expansion step on a 1-D Gaussian base.

    With p_t = N(0, V_t), base drift b = g(t)x and running reward
    f_t(x) = gamma*lambda_t*x^2/(2 V_t) (so grad f = -gamma*lambda_t*s_t), the
    Feynman-Kac solution is exactly h = exp(alpha(t) x^2/2 + beta(t)) with

        alpha' = -2 g alpha - sigma^2 alpha^2 - gamma*lambda/V,   alpha(1) = 0

    and the optimal drift correction is sigma_t^2 * alpha(t) * x. This is the
    ground truth the lean adjoint must reproduce: `a` should equal alpha(t)*x.

    Returns (ts descending from 1 to t_min, alpha).
    """
    import numpy as np

    ts = np.linspace(1.0, t_min, n)
    dt = ts[1] - ts[0]
    a, out = 0.0, np.empty(n)
    for i, t in enumerate(ts):
        out[i] = a
        tc = min(max(float(t), 1e-9), 1 - 1e-9)
        V = (1 - tc) ** 2 + tc**2 * std**2
        sig2 = 2 * (1 - tc) / tc
        g = 2 * (tc * std**2 - (1 - tc)) / V - 1 / tc
        a = a + dt * (-2 * g * a - sig2 * a * a - gamma * lam_fn(tc) / V)
    return ts, out
