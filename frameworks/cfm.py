import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFMTrajectory:
    times: torch.Tensor  # (steps + 1,)
    states: torch.Tensor  # (steps + 1, n, dim)
    dt: float


class CFM:
    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def loss(self, model: nn.Module, x1: torch.Tensor, x0: torch.Tensor = None):
        if x0 is None:
            x0 = torch.randn_like(x1)

        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        t_expanded = t.view(B, 1)
        xt = (1.0 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0

        vt = model(xt, t)
        return F.mse_loss(vt, ut)

    @staticmethod
    def score_from_velocity(velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """Eq. 19 for ``X_t = (1-t) X_0 + t X_1``."""
        return (t.reshape(-1, 1) * velocity - x) / (1 - t.reshape(-1, 1))

    @staticmethod
    def memoryless_noise_std(t: float):
        """Eq. 51's memoryless diffusion coefficient for the linear path."""
        return math.sqrt(2 * (1 - t) / t)

    def memoryless_sde_drift(self, model: nn.Module, x: torch.Tensor, t: float):
        """Eq. 51 drift: ``u + sigma(t)^2 score / 2 = 2u - x/t``."""
        time = torch.full((len(x),), t, device=x.device, dtype=x.dtype)
        return 2 * model(x, time) - x / t

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        n: int = None,
        dim: int = None,
        num_steps: int = 50,
        x0: torch.Tensor = None,
        full: bool = False,
        memoryless_sde: bool = False,
        t_min: float = 0.05,
    ):
        model.eval()
        if dim is None:
            dim = model.dim
        device = next(model.parameters(), torch.empty(0)).device
        if x0 is None:
            if n is None:
                raise ValueError("sample() needs either n or x0")
            x0 = torch.randn(n, dim, device=device)
        else:
            x0 = x0.to(device)

        if memoryless_sde:
            trajectory = self._sample_memoryless_sde(model, x0, num_steps, t_min)
            return trajectory if full else trajectory.states[-1]

        # Midpoint ODE method used by the existing CFM sampler.
        x = x0
        n = x.shape[0]
        dt = 1.0 / num_steps
        states = [x] if full else None

        for i in range(num_steps):
            t_val = i * dt
            t_tensor = torch.full((n,), t_val, device=device, dtype=x.dtype)

            v1 = model(x, t_tensor)
            x_mid = x + 0.5 * dt * v1
            t_mid = torch.full((n,), t_val + 0.5 * dt, device=device, dtype=x.dtype)
            v_mid = model(x_mid, t_mid)

            x = x + dt * v_mid
            if full:
                states.append(x)

        if full:
            return CFMTrajectory(
                times=torch.linspace(0, 1, num_steps + 1, device=device),
                states=torch.stack(states),
                dt=dt,
            )

        return x

    @torch.no_grad()
    def _sample_memoryless_sde(
        self, model: nn.Module, x0: torch.Tensor, num_steps: int, t_min: float
    ):
        """Algorithm 4's full memoryless-SDE trajectory."""
        x = x0
        n = len(x)
        dt = (1 - t_min) / num_steps
        times = torch.linspace(t_min, 1, num_steps + 1, device=x.device, dtype=x.dtype)

        warm_dt = t_min / 5
        for warm_step in range(5):
            time = torch.full((n,), warm_step * warm_dt, device=x.device, dtype=x.dtype)
            x = x + warm_dt * model(x, time)

        states = [x]
        for time in times[:-1]:
            t = float(time)
            x = x + dt * self.memoryless_sde_drift(model, x, t)
            x = x + math.sqrt(dt) * self.memoryless_noise_std(t) * torch.randn_like(x)
            states.append(x)
        return CFMTrajectory(times=times, states=torch.stack(states), dt=dt)


class ReFlow(CFM):
    def __init__(
        self, teacher, teacher_model, num_gen_steps: int = 50, sigma_min: float = 1e-4
    ):
        super().__init__(sigma_min=sigma_min)
        self.teacher = teacher
        self.teacher_model = teacher_model
        self.num_gen_steps = num_gen_steps

    def loss(self, model, x_0):
        n, dim = x_0.shape[0], x_0.shape[1]
        noise = torch.randn(n, dim, device=x_0.device)
        synthetic_x1 = self.teacher.sample(
            self.teacher_model, x0=noise, num_steps=self.num_gen_steps
        )
        return super().loss(model, synthetic_x1, x0=noise)
