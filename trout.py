#!/usr/bin/python3

import sys
import argparse
import subprocess
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
        self.model = nn.Sequential(*layers)

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
    
    
input_dim = 30
hl1 = 32
hl2 = 64
hl3 = 32
dropout = 0.15
activ = "elu"

model_state_dict_path = "best_model.pt"
    
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", dest="job", default=-1, help="Job ID")
args = parser.parse_args()

if __name__=="__main__":
    
    if args.job < 0:
        print("Must provide job id with -j or --job flag")
        sys.exit(-1)
        
    # Get input variables
    # Job Specific Features
    sacct_command = f"sacct -j {args.job} -X -o Partition,TimelimitRaw,Priority,ReqCPUS,ReqMem,ReqNodes --parsable2"
    result = subprocess.run([sacct_command], shell=True, capture_output=True, text=True).stdout.split("|")
    
    
    
    # Queue Features
    
    
    
    
   ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
                   # "day_of_week", "day_of_year",
                    "par_jobs_ahead_queue", "par_cpus_ahead_queue", "par_memory_ahead_queue",
                    "par_nodes_ahead_queue", "par_time_limit_ahead_queue",
                    "par_jobs_queue", "par_cpus_queue", "par_memory_queue",
                    "par_nodes_queue", "par_time_limit_queue",
                    "par_jobs_running", "par_cpus_running", "par_memory_running",
                    "par_nodes_running", "par_time_limit_running",
                    "user_jobs_past_day", "user_cpus_past_day",
                    "user_memory_past_day", "user_nodes_past_day",
                    "user_time_limit_past_day",
                    "par_total_nodes", "par_total_cpu", "par_cpu_per_node", "par_mem_per_node", "par_total_gpu"] 
    
    
    
    model = nn_model(input_dim, hl1, hl2, hl3, dropout, activ)
    model.load_state_dict(model_state_dict_path)
    
    
    
    
    
    
    