import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

from utils import Sin


class PINN(nn.Module):
    def __init__(self, input_size):
        super(PINN, self).__init__()

        self._build_layers(
            input_size,
        )
        self._set_optimizer()

    def _build_layers(self, input_size):
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            Sin(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def _nth_deriv(self, X_batch, y_pred, idx, n):
        var = X_batch[:, idx : idx + 1]

        for _ in range(n):
            y_pred = torch.autograd.grad(
                outputs=y_pred,
                inputs=var,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True,
            )[0]

        return y_pred

    def _get_physics_loss(self, X_batch, y_pred):
        # To satisfy dy/dx >= 0
        return torch.mean(torch.relu(-self._nth_deriv(X_batch, y_pred, 0, 1)) ** 2)

    def loss_fn(self, X_batch, y_batch):
        X_batch = X_batch.requires_grad_(True)
        pred = self.forward(X_batch)

        data_loss = F.mse_loss(pred, y_batch)
        physics_loss_weight = 1
        physics_loss = self._get_physics_loss(X_batch, pred)

        return data_loss + physics_loss_weight * physics_loss

    def forward(self, x):
        return self.layers(x)
