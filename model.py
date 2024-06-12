import torch
import torch.nn as nn


class nn_model(nn.Module):
    def __init__(self, input_dim, hl1, hl2, dropout):
        super(nn_model, self).__init__()
        activation_layer = nn.ReLU()
        layers = []
        layers.append(nn.Linear(input_dim, hl1))
        layers.append(activation_layer)
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hl1, hl2))
        layers.append(activation_layer)
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hl2, 32))
        layers.append(activation_layer)
        layers.append(nn.Linear(32, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out