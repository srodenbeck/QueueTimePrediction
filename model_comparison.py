# -*- coding: utf-8 -*-

from model import nn_model
import read_db
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from absl import app, flags
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

custom_loss = {
    "train_within_10min_correct":  0,
    "train_within_10min_total":  0,
    "test_within_10min_correct":  0,
    "test_within_10min_total":  0,
    "train_binary_10min_correct":  0,
    "train_binary_10min_total":  0,
    "test_binary_10min_correct":  0,
    "test_binary_10min_total":  0,
    "val_within_10min_correct": 0,
    "val_within_10min_total": 0,
    "val_binary_10min_correct": 0,
    "val_binary_10min_total": 0
}

def calculate_custom_loss(pred, y, train_test_val):
    """
    calculate_custom_loss()

    Updates custom_loss with various losses from custom loss functions,
    including the accuracy of predictions within 10 minutes and the accuracy
    of the model as a binary classifier with a split point at 10 minutes.

    Parameters
    ----------
    pred : float
        Predicted value from the model.
    y : float
        Actual value which model attempts to predict..
    train_test_val : TYPE
        What part of the data is passed in.

    Returns
    -------
    None.

    """
    # Calculating loss in regards to how many target values were within 10 
    # minutes of predicted values
    sub_tensor = torch.sub(pred.flatten(), y)
    binary_within = torch.where(torch.abs(sub_tensor) < 10, 1, 0)
    pred[pred < 0] = 0
    custom_loss[f"{train_test_val}_within_10min_total"] += binary_within.shape[0]
    custom_loss[f"{train_test_val}_within_10min_correct"] += binary_within.sum().item()

    # Calculating loss in regards to number of correct binary classifications,
    # splitting data at 10 minutes
    y_binary = torch.where(y > 10, 1, 0)
    pred_binary = torch.where(pred.flatten() > 10, 1, 0)
    binary_10min = torch.where(y_binary == pred_binary, 1, 0)
    custom_loss[f"{train_test_val}_binary_10min_total"] += binary_10min.shape[0]
    custom_loss[f"{train_test_val}_binary_10min_correct"] += binary_10min.sum().item()

def get_planned_target_index(df):
    """
    get_planned_target_index()
    Returns the index of the column 'planned' in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of information.

    Returns
    -------
    INT
        Index of the 'planned' column in df.

    """
    return df.columns.get_loc('planned')


def get_feature_indices(df, feature_names):
    """
    get_feature_indices()

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of information from database.
    feature_names : List (str)
        List of strings containing the names of desired features.

    Returns
    -------
    feature_indices : List (int)
        List of indices of all features in feature_names from df.

    """
    feature_indices = []
    for feature_name in feature_names:
        try:
            feature_indices.append(df.columns.get_loc(feature_name))
        except Exception as e:
            print(f"Error: Could not find '{feature_name}' in database\nExiting...")
            sys.exit(1)
    return feature_indices



FLAGS = flags.FLAGS
print("Reading in table")

engine = read_db.create_engine()
table = "jobs_all_2"
df = pd.read_sql_query(f"SELECT * FROM {table} WHERE submit >= '2024-04-01'", engine)

print("Read in table")
print(df.shape)



feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem",
                 "jobs_ahead_queue", "jobs_running", "cpus_ahead_queue",
                 "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
                 "cpus_running", "memory_running", "nodes_running", "time_limit_running",
                 'jobs_ahead_queue_priority', 'cpus_ahead_queue_priority',
                 'memory_ahead_queue_priority', 'nodes_ahead_queue_priority',
                 'time_limit_ahead_queue_priority']
                 # "partition", "qos"]
                 
num_features = len(feature_names)
print(feature_names)
print(num_features)

if FLAGS.only_10min_plus:
    print("10 plus flag: ", FLAGS.only_10min_plus)
    df = df[df['planned'] > 10 * 60]
    print(f"Using {len(df)} jobs")
    
np_array = df.to_numpy()

# Read in desired features and target columns to numpy arrays
feature_indices = get_feature_indices(df, feature_names)
target_index = get_planned_target_index(df)
X, y = np_array[:, feature_indices], np_array[:, target_index]
X = X.astype(np.float32)
y = y.astype(np.float32)

# Transformations
y = y / 60

if FLAGS.transform:
     # X_train, X_test = transformations.scale_min_max(X_train, X_test)
     X = np.log(X + 1)

if FLAGS.transform_target:
   y = np.log(y + 1)
     
X_to_tensor = torch.from_numpy(X).to(torch.float32)
y_to_tensor = torch.from_numpy(y).to(torch.float32)

dataset = TensorDataset(X_to_tensor, y_to_tensor)
dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size)

print("Data finished transformations")

PATH = "regr_model.pt"
# Create model
model = nn_model(num_features, FLAGS.hl1, FLAGS.hl2, FLAGS.dropout, FLAGS.activ)
model.load_state_dict(torch.load(PATH))

if FLAGS.loss == "l1_loss":
    loss_fn = nn.L1Loss
elif FLAGS.loss == "mse_loss":
    loss_fn = nn.MSELoss()
elif FLAGS.loss == "smooth_l1_loss":
    loss_fn = nn.SmoothL1Loss()
else:
    sys.exit(f"Loss function '{FLAGS.loss}' not supported")


model.eval()
y_pred = []
y_actual= []
test_loss = []
absolute_percentage_error = []
with torch.no_grad():
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)
        test_loss.append(loss.item())
        
        flat_pred = pred.flatten()
        for idx in range(len(y)):
            ape = abs((y[idx] - flat_pred[idx]) / y[idx])
            absolute_percentage_error.append(ape)
        
        calculate_custom_loss(pred.flatten(), y, "test")
        y_pred.extend(pred.flatten())
        y_actual.extend(y)
avg_test_loss = np.mean(np.mean(test_loss))
print(f"Average test loss of {avg_test_loss}")
bin_width = 0.1
bins = np.arange(min(absolute_percentage_error), max(absolute_percentage_error) + bin_width, bin_width)
plt.hist(absolute_percentage_error, bins=bins, density=True)
plt.xlabel('Absolute Percentage Error')
plt.ylabel('Probability Density')
plt.title("Density Histogram of Absolute Percentage Error of Test Data")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.show()

# Average absolute percentage error
avg_ape = np.mean(absolute_percentage_error)
print("Mean absolute percentage error:", avg_ape * 100, "%")

r_value = pearsonr(y_pred, y_actual)
print("pearson's r: ", r_value)
plt.scatter(y_pred, y_actual)
plt.xlabel("y predicted")
plt.ylabel("y")
plt.show()


