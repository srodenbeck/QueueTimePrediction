import torch
import torch.nn as nn


class nn_model(nn.Module):
    def __init__(self, input_dim, hl1, hl2, dropout, activ):
        super(nn_model, self).__init__()
        if activ == "leaky_relu":
            activation_layer = nn.LeakyReLU()
        elif activ == "elu":
            activation_layer = nn.ELU()
        else:
            activation_layer = nn.ReLU()

        layers = [
            nn.Linear(input_dim, hl1),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hl1, hl2),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hl2, 32),
            activation_layer,
            nn.Linear(hl2, 1)
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

class classify_model(nn.Module):
    def __init__(self, input_dim, hl1, hl2):
        super(classify_model, self).__init__()
        layers = [
            nn.Linear(input_dim, hl1),
            nn.ReLU(),
            nn.Linear(hl1, hl2),
            nn.ReLU(),
            nn.Linear(hl2, 2)
        ]
        self.model = torch.nn.Sequential(*layers)


    def forward(self, x):
        out = self.model(x)
        return out