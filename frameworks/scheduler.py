import torch


class DDPMScheduler:
    def __init__(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(T)

    def to(self, device):
        """Move scheduler coefficients alongside the DDPM model."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def respace(self, timesteps):
        alpha_bars = self.alpha_bars[timesteps]
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        self.alpha_bars = alpha_bars
        self.alphas = alpha_bars / alpha_bars_prev
        self.betas = 1.0 - self.alphas
        self.timesteps = timesteps
        self.T = timesteps.numel()

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bars[t].view(-1, *([1] * (x_0.ndim - 1)))

        mean = torch.sqrt(alpha_bar_t) * x_0
        var = 1.0 - alpha_bar_t
        x_t = mean + torch.sqrt(var) * noise
        return x_t, noise

    def reverse_indices(self, j):
        """Algorithm-2 index ``j`` -> (scheduler chain index, denoiser label)."""
        chain_index = self.T - 1 - j
        labels = getattr(
            self, "timesteps", torch.arange(self.T, device=self.alpha_bars.device)
        )
        label_index = (
            chain_index.to(labels.device)
            if torch.is_tensor(chain_index)
            else chain_index
        )
        return chain_index, labels[label_index]

    def step_mean(self, model_output, t, x_t):
        beta_t = self.betas[t].to(x_t)
        alpha_t = self.alphas[t].to(x_t)
        alpha_bar_t = self.alpha_bars[t].to(x_t)
        return (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(
            alpha_t
        )

    def step(self, model_output, t, x_t):
        mean = self.step_mean(model_output, t, x_t)
        if t == 0:
            return mean
        beta_t = self.betas[t].to(x_t)
        alpha_bar_t = self.alpha_bars[t].to(x_t)
        alpha_bar_prev = self.alpha_bars[t - 1].to(x_t)
        var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)
