import torch
import torch.nn as nn


class nn_model(nn.Module):
    def __init__(self, input_dim, hl1, hl2):
        super(nn_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hl1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hl1, hl2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hl2, 1)
        )

    def forward(self, x):
        out = self.layers(x)
        return out