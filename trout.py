#!/usr/bin/python3

import sys
import argparse
import subprocess
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime, timedelta
import joblib
from torch import load

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
    
    
def convert_minutes(total_minutes):
    days = total_minutes // (24 * 60)
    hours = (total_minutes % (24 * 60)) // 60
    minutes = total_minutes % 60
    return days, hours, minutes
    
def memory_to_gigabytes(memory_str):
    """
    memory_to_gigabytes
    
    Parameters
    ----------
    memory_str : STRING
        String in format xxxxU where xxxx is a number and U is a character
        representing the unit of memory used.

    Raises
    ------
    ValueError
        Unit of measurement is not recognized.

    Returns
    -------
    FLOAT
        Returns a float representing the memory of memory_str converted to
        gigabytes.

    """
    if memory_str.endswith('T'):
        return float(memory_str[:-1]) * 1024
    elif memory_str.endswith('G'):
        return float(memory_str[:-1])
    elif memory_str.endswith('M'):
        return float(memory_str[:-1]) / 1024
    elif memory_str == "0":
        return float(0)
    else:
        raise ValueError(f"Unknown memory unit in {memory_str}")
        

def add_minutes_to_current_time(minutes_to_add):
    # Get the current time
    current_time = datetime.now()
    
    # Create a timedelta object with the specified minutes
    time_to_add = timedelta(minutes=minutes_to_add)
    
    # Add the timedelta to the current time
    new_time = current_time + time_to_add
    
    # Return new time
    return new_time.strftime("%Y-%m-%d %H:%M:%S")

def make_input_array(result, queue_ahead_df, queue_df, running_df, user_past_day_df, pred_run_time):
    # Priority
    feature_vals.append(result[2])
    # TimelimitRaw
    feature_vals.append(result[1])
    # ReqCPUS
    feature_vals.append(result[3])
    # ReqMem
    feature_vals.append(memory_to_gigabytes(result[4]))
    # ReqNodes
    feature_vals.append(result[5])
    
    # par jobs ahead queue
    feature_vals.append(len(queue_ahead_df))
    # par cpus ahead queue
    feature_vals.append(sum(queue_ahead_df["ReqCPUS"]))
    # par memory ahead queue
    feature_vals.append(sum(queue_ahead_df["ReqMem"]))
    # par nodes ahead queue
    feature_vals.append(sum(queue_ahead_df["ReqNodes"]))
    # par time limit ahead queue
    feature_vals.append(sum(queue_ahead_df["TimelimitRaw"]))
    
    # par jobs queue
    feature_vals.append(len(queue_df))
    # par cpus queue
    feature_vals.append(sum(queue_df["ReqCPUS"]))
    # par memory queue
    feature_vals.append(sum(queue_df["ReqMem"]))
    # par nodes queue
    feature_vals.append(sum(queue_df["ReqNodes"]))
    # par time limit queue
    feature_vals.append(sum(queue_df["TimelimitRaw"]))
    
    # par jobs running
    feature_vals.append(len(running_df))
    # par cpus running
    feature_vals.append(sum(running_df["ReqCPUS"]))
    # par memory running
    feature_vals.append(sum(running_df["ReqMem"]))
    # par nodes running
    feature_vals.append(sum(running_df["ReqNodes"]))
    # par time limit running
    feature_vals.append(sum(running_df["TimelimitRaw"]))
    
    # user jobs past day
    feature_vals.append(len(user_past_day_df))
    # user cpus past day
    feature_vals.append(sum(user_past_day_df["ReqCPUS"]))
    # user memory past day
    feature_vals.append(sum(user_past_day_df["ReqMem"]))
    # user nodes past day
    feature_vals.append(sum(user_past_day_df["ReqNodes"]))
    # user time limit past day
    feature_vals.append(sum(user_past_day_df["TimelimitRaw"]))
    
    # par total nodes
    feature_vals.append(partition_feature_dict[PARTITION][0])
    # par total cpus
    feature_vals.append(partition_feature_dict[PARTITION][1])
    # par cores/node
    feature_vals.append(partition_feature_dict[PARTITION][2])
    # par mem/node
    feature_vals.append(partition_feature_dict[PARTITION][3])
    # par total gpus
    feature_vals.append(partition_feature_dict[PARTITION][4])
    
    # pred run time
    feature_vals.append(pred_run_time)
    # par queue time limit pred
    feature_vals.append(sum(queue_df["PredRuntime"]))
    # par pred timelimit running
    feature_vals.append(sum(running_df["PredRuntime"]))

    input_arr = np.array(feature_vals)

    input_arr = np.log1p(input_arr)
    
    return input_arr



input_dim = 33
hl1 = 32
hl2 = 64
hl3 = 32
dropout = 0.15
activ = "elu"

# Format is - partition: [#Nodes, #CPU_cores, Cores/Node, Mem/Node (GB), GPU]
partition_feature_dict = {
    'wholenode': [750, 96000, 128, 257, 0],
    'standard': [750, 96000, 128, 257, 0],
    'shared': [250, 32000, 128, 257, 0],
    'wide': [750, 96000, 128, 257, 0],
    'highmem': [32, 4096, 128, 1031, 0],
    'debug': [17, 2176, 128, 257, 0],
    'gpu-debug': [16, 2048, 128, 515, 64],
    'benchmarking': [1048, 134144, 128, 257, 0],
    'azure': [8, 16, 2, 7, 0],
    'gpu': [16, 2048, 128, 515, 64]
}


model_state_dict_path = "best_model.pt"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job", dest="job", default=-1, help="Job ID")
    args = parser.parse_args()
    
    
    # Input to saved nn model
    feature_vals = []
    
    if int(args.job) < 0:
        print("Must provide job id with -j or --job flag")
        sys.exit(-1)
    
    # Get input variables
    # Job Specific Features
    
    #TODO: add in -S to account for a day (or potentially longer, rather than 00:00:00 of current day)
    # start_time = 
        
#         Note, valid start/end time formats are...
# 	           HH:MM[:SS] [AM|PM]
# 	           MMDD[YY] or MM/DD[/YY] or MM.DD[.YY]
# 	           MM/DD[/YY]-HH:MM[:SS]
# 	           YYYY-MM-DD[THH:MM[:SS]]
# 	           now[{+|-}count[seconds(default)|minutes|hours|days|weeks]]
        
    sacct_command = f"sacct -j {int(args.job)} -X -o Partition,TimelimitRaw,Priority,ReqCPUS,ReqMem,ReqNodes,User --parsable2"
    result = subprocess.run([sacct_command], shell=True, capture_output=True, text=True).stdout.split("\n")[1].split("|")

    PARTITION = result[0]
    par_sacct_command = f"sacct -a -X -r {PARTITION} -o TimelimitRaw,Priority,ReqCPUS,ReqMem,ReqNodes,State --parsable2"
    par_result = subprocess.run([par_sacct_command], shell=True, capture_output=True, text=True).stdout.split("\n")

    USERNAME = result[6]
    user_sacct_command = f"sacct -u {USERNAME} -S $(date -d '1 day ago' +%Y-%m-%dT%H:%M:%S) -o JobID,User,ReqCPUS,ReqMem,ReqNodes,TimelimitRaw,State --parsable2"
    user_past_day_result = subprocess.run([user_sacct_command], shell=True, capture_output=True, text=True).stdout.split("\n")

    col_names = par_result[0].split("|")
    split_data = [item.split("|") for item in par_result[1:-1]]
    df = pd.DataFrame(split_data, columns=col_names).astype(str)
    df.replace("None", "0")

    df['TimelimitRaw'] = df['TimelimitRaw'].replace('', '0') 
    df['TimelimitRaw'] = df['TimelimitRaw'].astype(int)
    df['Priority'] = df['Priority'].astype(int)
    df['ReqCPUS'] = df['ReqCPUS'].astype(int)
    df['ReqNodes'] = df['ReqNodes'].astype(int)
    df["ReqMem"] = df["ReqMem"].apply(memory_to_gigabytes)
    df["ReqMem"] = df["ReqMem"].astype(int)
 
    col_names = user_past_day_result[0].split("|")
    split_data = [item.split("|") for item in user_past_day_result[1:-1]]
    user_past_day_df = pd.DataFrame(split_data, columns=col_names)
    
 
    # [#Nodes, #CPU_cores, Cores/Node, Mem/Node (GB), GPU]
    df["par_total_nodes"] = partition_feature_dict[result[0]][0]
    df["par_total_cpu"] = partition_feature_dict[result[0]][1]
    df["par_cpu_per_node"] = partition_feature_dict[result[0]][2]
    df["par_mem_per_node"] = partition_feature_dict[result[0]][3]
    df["par_total_gpu"] = partition_feature_dict[result[0]][4]
    
    
    # TODO: Potentially use
    # features for pred run time = ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
    #                  "par_total_nodes", "par_total_cpu", "par_cpu_per_node", "par_mem_per_node", "par_total_gpu"]
    
    reg = joblib.load('pred_runtime.joblib')

    df["PredRuntime"] = reg.predict(df[["Priority", "TimelimitRaw", "ReqCPUS",
                                        "ReqMem", "ReqNodes", "par_total_nodes",
                                        "par_total_cpu", "par_cpu_per_node",
                                        "par_mem_per_node", "par_total_gpu"]])
    
    main_pred_runtime = reg.predict([[result[2], result[1], result[3], memory_to_gigabytes(result[4]), result[5],
                                     partition_feature_dict[result[0]][0],
                                     partition_feature_dict[result[0]][1],
                                     partition_feature_dict[result[0]][2],
                                     partition_feature_dict[result[0]][3],
                                     partition_feature_dict[result[0]][4]
                                     ]])
      
    running_df = df[df["State"] == "RUNNING"]
    queue_df = df[df["State"] == "PENDING"]
    queue_ahead_df = queue_df[queue_df["Priority"] > int(result[2])]
    user_past_day_df= user_past_day_df[user_past_day_df["State"] == "RUNNING"]
    user_past_day_df = user_past_day_df[user_past_day_df["State"] == "COMPLETED"]
        
    # Make input array from dataframes and result
    input_arr = make_input_array(result, queue_ahead_df, queue_df, running_df, user_past_day_df, main_pred_runtime)
    input_dim = len(input_arr)

    # Loading and running model
    model = nn_model(input_dim, hl1, hl2, hl3, dropout, activ)
    model.load_state_dict(load(model_state_dict_path))
    model.eval()
    
    # Pred is estimated time in minutes until start
    pred = model(input_arr)
    pred = round(pred)
    
    d, h, m = convert_minutes(pred)
    estimated_start_time = add_minutes_to_current_time(pred)
    print(f"Job {args.job} is estimated to start in {d} day(s), {h} hour(s), and {m} minute(s) ({estimated_start_time})")
    
    
    
    
    
   
                    
      
    
