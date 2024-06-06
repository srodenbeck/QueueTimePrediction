import torch
import torch.nn as nn


class nn_model(nn.Module):
    def __init__(self, input_dim):
        super(nn_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = self.layers(x)
        return out