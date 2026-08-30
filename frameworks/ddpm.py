import torch
import torch.nn.functional as F


class DDPM:
    "Basic DDPM implementation with uniform grid."

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def loss(self, model, x_0):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.scheduler.T, (batch_size,), device=x_0.device)
        x_t, noise = self.scheduler.add_noise(x_0, t)
        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model, n, dim=None, full=False):
        if dim is None:
            dim = model.dim

        timesteps = getattr(self.scheduler, "timesteps", torch.arange(self.scheduler.T))

        device = next(model.parameters(), torch.empty(0)).device
        x = torch.randn(n, dim, device=device)

        traj = torch.empty(self.scheduler.T + 1, n, dim, device=device)
        traj[0] = x

        for j, i in enumerate(range(self.scheduler.T - 1, -1, -1)):
            t_orig = int(timesteps[i])
            t_batch = torch.full((n,), t_orig, dtype=torch.long, device=device)
            predicted_noise = model(x, t_batch)
            x = self.scheduler.step(predicted_noise, i, x)
            traj[j + 1] = x

        return traj if full else x


# Only used for sampling so no loss fn.
class DDIM:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(self, model, n, dim=None):
        if dim is None:
            dim = model.dim

        alpha_bars = self.scheduler.alpha_bars
        x = torch.randn(n, dim)
        for t in range(self.scheduler.T - 1, -1, -1):
            t_batch = torch.full((n,), t, dtype=torch.long)
            eps = model(x, t_batch)

            alpha_bar_t = alpha_bars[t]
            alpha_bar_prev = (
                alpha_bars[t - 1] if t > 0 else torch.ones_like(alpha_bar_t)
            )

            x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x = (
                torch.sqrt(alpha_bar_prev) * x_0_pred
                + torch.sqrt(1 - alpha_bar_prev) * eps
            )
        return x
