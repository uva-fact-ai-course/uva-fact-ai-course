import torch
import torch.nn as nn


class VarDropout(nn.Module):
    """Variational dropout.

    Keeps single dropout mask during forward pass, for all timesteps.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout):
        if not self.training or not dropout:
            return x
        mask = torch.zeros(1, x.size(1), x.size(2), requires_grad=False, device=x.device).bernoulli_(1 - dropout) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
