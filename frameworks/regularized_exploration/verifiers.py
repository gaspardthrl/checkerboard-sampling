import torch

__all__ = ["checkerboard_verifier", "grad_of"]


def grad_of(log_v):
    """Autograd wrapper turning a scalar-per-point log_v into grad_log_v."""

    def grad_log_v(x: torch.Tensor):
        xx = x.detach().requires_grad_(True)
        with torch.enable_grad():
            out = log_v(xx).sum()
        (g,) = torch.autograd.grad(out, xx)
        return g.detach()

    return grad_log_v


def checkerboard_verifier(n_tiles, tau=4.0, dilate=0.0, box=1.0, margin=0.03):
    cell = 2 * box / n_tiles

    # Construct valid tile boxes, slightly eroded.
    boxes = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            if (i + j) % 2 == 0:
                lo = torch.tensor(
                    [
                        -box + i * cell + margin,
                        -box + j * cell + margin,
                    ]
                )
                hi = torch.tensor(
                    [
                        -box + (i + 1) * cell - margin,
                        -box + (j + 1) * cell - margin,
                    ]
                )
                boxes.append((lo, hi))

    def log_v(x):
        distances = []

        for lo, hi in boxes:
            lo, hi = lo.to(x), hi.to(x)

            # Euclidean distance to this shrunken valid box.
            delta = torch.maximum(
                torch.maximum(lo - x, x - hi),
                torch.zeros_like(x),
            )
            distances.append(delta.square().sum(-1))

        dist2 = torch.stack(distances, dim=-1).min(dim=-1).values

        # Don't penalize points already accepted by the true verifier.
        valid = hard(x)
        return torch.where(valid, torch.zeros_like(dist2), -tau * dist2)

    def hard(x):
        inside = (x.abs() <= box).all(dim=-1)
        u = (x + box) / cell
        idx = torch.floor(u).long().clamp(0, n_tiles - 1)
        return inside & ((idx.sum(dim=-1) % 2) == 0)

    return {
        "log_v": log_v,
        "grad_log_v": grad_of(log_v),
        "hard": hard,
    }
