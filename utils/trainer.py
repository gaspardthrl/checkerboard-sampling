import numpy as np
import torch

from utils.normalize import to_model_space


def train(model, framework, dataset, num_steps, batch_size=256, lr=1e-3, log_every=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    for step in range(num_steps):
        x_0 = dataset.sample(batch_size, rng)
        x_0 = to_model_space(torch.from_numpy(x_0).float()).to(device)

        optimizer.zero_grad()
        loss = framework.loss(model, x_0)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_steps - 1:
            print(f"step {step:6d} | loss {loss.item():.4f}")

    return model
