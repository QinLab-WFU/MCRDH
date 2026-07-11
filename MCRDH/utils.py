import numpy as np
import torch
from torch import nn


class MovingAverageStrategy(nn.Module):
    """
    adapted from paper: Ensemble of Loss Functions to Improve Generalizability of Deep Metric Learning methods
    """

    def __init__(self, prior, eta=10000):
        super().__init__()
        self.eta = eta
        self.n_losses = len(prior)

        # 𝛼 maintains a minimum weight for each loss function
        self.alpha = 1 / (4 * self.n_losses)

        c = torch.from_numpy(prior / np.sum(prior))  # - self.alpha
        self.c = nn.Parameter(c.sqrt(), requires_grad=True)

        self.l_ma = np.zeros(self.n_losses)  # mean of losses obtained by exp moving average

        self.iter = 0

    def forward(self, l):
        """
        Args:
            l: losses array
        Returns:
            final loss follow Eq. 11 of the paper
        """

        for i in range(self.n_losses):
            # l_ma[j]记录lj的历史平均值: lj‾
            # l_i = l[i].item() if isinstance(l[i], torch.Tensor) else l[i]
            self.l_ma[i] = (l[i].item() + self.iter * self.l_ma[i]) / (self.iter + 1)

        self.iter += 1

        # l‾
        l_mean = np.mean(self.l_ma)

        c2 = self.c**2

        final_loss_part1 = 0
        for i in range(self.n_losses):
            l_hat = (l_mean / self.l_ma[i]) * l[i]  # Eq. 8
            final_loss_part1 += (c2[i] + self.alpha) * l_hat  # first part of Eq. 11

        final_loss_part2 = self.eta * (c2.sum() + self.n_losses * self.alpha - 1) ** 2

        return final_loss_part1 + final_loss_part2
