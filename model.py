import torch
import torch.nn as nn


class nn_model(nn.Module):
    def __init__(self, input_dim, hl1, hl2, hl3, dropout, activ):
        """
        __init__()
        
        Initiates nn_model instance with given hyperparameters

        Parameters
        ----------
        input_dim : int
            Size of input data.
        hl1 : int
            Size of first hidden layer.
        hl2 : int
            Size of second hidden layer.
        dropout : float
            Proportion of data to drop (ranging from 0.0 to 1.0).
        activ : function
            Activation function to use. Base case is ReLU function.

        Returns
        -------
        None.

        """
        super(nn_model, self).__init__()
        if activ == "leaky_relu":
            activation_layer = nn.LeakyReLU()
        elif activ == "elu":
            activation_layer = nn.ELU()
        else:
            activation_layer = nn.ReLU()

        layers = [
            nn.Linear(input_dim, hl1),
            nn.BatchNorm1d(hl1),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hl1, hl2),
            nn.BatchNorm1d(hl2),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hl2, hl3),
            nn.BatchNorm1d(hl3),
            activation_layer,
            nn.Linear(hl3, 1)
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        forward()
        
        Feeds input data x through the model's layers

        Parameters
        ----------
        x : Array
            Input data stored in an array format.

        Returns
        -------
        out : Float
            Predicted value from passing x through the model.

        """
        out = self.model(x)
        return out
    
    
    
    
class job_model(nn.Module):
    def __init__(self, input_dim, hl1, hl2, dropout, activ):
        super(job_model, self).__init__()
        if activ == "leaky_relu":
            activation_layer = nn.LeakyReLU()
        elif activ == "elu":
            activation_layer = nn.ELU()
        else:
            activation_layer = nn.ReLU()

        layers = [
            nn.Linear(input_dim, hl1),
            # nn.BatchNorm1d(hl1),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hl1, hl2),
            # nn.BatchNorm1d(hl2),
            activation_layer,
            nn.Linear(hl2, 1)
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

