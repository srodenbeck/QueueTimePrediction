# -*- coding: utf-8 -*-

from absl import app, flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import neptune
import itertools

import config_file
import read_db
from model import nn_model


# flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
# flags.DEFINE_float('lr', 0.01, 'Learning rate.')
# flags.DEFINE_integer('batch_size', 32, 'Batch size')
# flags.DEFINE_integer('epochs', 100, 'Number of Epochs')
# flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
# flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
# flags.DEFINE_integer('hl1', 32, 'Hidden layer 1 dim')
# flags.DEFINE_integer('hl2', 24, 'Hidden layer 1 dim')
# flags.DEFINE_float('dropout', 0.2, 'Dropout rate')

FLAGS = flags.FLAGS

def get_planned_target_index(df):
    return df.columns.get_loc('planned')

def get_feature_indices(df, feature_names):
    feature_indices = []
    for feature_name in feature_names:
        try:
            feature_indices.append(df.columns.get_loc(feature_name))
        except Exception as e:
            print(f"Error: Could not find '{feature_name}' in database\nExiting...")
            sys.exit(1)
    return feature_indices


def create_dataloaders(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    X_train, X_test = scale_min_max(X_train, X_test)

    # First step: converting to tensor
    x_train_to_tensor = torch.from_numpy(X_train).to(torch.float32)
    y_train_to_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
    y_test_to_tensor = torch.from_numpy(y_test).to(torch.float32)

    # Second step: Creating TensorDataset for Dataloader
    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    test_dataset = TensorDataset(x_test_to_tensor, y_test_to_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size)
    return train_dataloader, test_dataloader

def scale_min_max(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def train(params, epochs, train_dataloader, val_dataloader):
    run = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
    )
    
    print(params)
    
    model = nn_model(params['num_features'], params['hl1'], params['hl2'], FLAGS.dropout)
    
    run["parameters"] = params
    
    # loss function
    if params['loss_fn'] == "l1_loss":
        loss_fn = nn.L1Loss
    elif params['loss_fn'] == "mse_loss":
        loss_fn = nn.MSELoss()
    elif params['loss_fn'] == "smooth_l1_loss":
        loss_fn = nn.SmoothL1Loss()
    else:
        sys.exit(f"Loss function '{params['loss_fn']}' not supported")

    # Optimizer
    if params['optimizer'] == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=FLAGS.lr)
    elif params['optimizer'] == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=FLAGS.lr)
    elif params['optimizer'] == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=FLAGS.lr)
    else:
        sys.exit(f"Optimizer '{params['optimizer']}' not supported")
        
    # Run training loop
    train_loss_by_epoch = []
    test_loss_by_epoch = []
    binary_10min_correct = 0
    binary_10min_total = 0
    final_loss = 0
    
    for epoch in range(epochs):
        train_loss = []
        test_loss = []
        train_within_10min_correct = 0
        train_within_10min_total = 0
        test_within_10min_correct = 0
        test_within_10min_total = 0
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())

            pred[pred < 0] = 0
            sub_tensor = torch.sub(pred.flatten(), y)
            binary_within = torch.where(torch.abs(sub_tensor) < 10, 1, 0)
            train_within_10min_total += binary_within.shape[0]
            train_within_10min_correct += binary_within.sum().item()

        for X, y in val_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                final_loss = loss
                # if epoch == epochs - 1:
                #     for i in range(y.shape[0]):
                #         print(f"Predicted: {pred.flatten()[i]} -- Real: {y[i]}")
                test_loss.append(loss.item())
                pred[pred < 0] = 0
                sub_tensor = torch.sub(pred.flatten(), y)
                binary_within = torch.where(torch.abs(sub_tensor) < 5, 1, 0)
                test_within_10min_total += binary_within.shape[0]
                test_within_10min_correct += binary_within.sum().item()

        print(f"Epoch = {epoch}, Train_loss = {np.mean(train_loss):.2f}, Test Loss = {np.mean(test_loss):.5f}")
        train_loss_by_epoch.append(np.mean(train_loss))
        test_loss_by_epoch.append(np.mean(test_loss))
        run["train/loss"].append(np.mean(train_loss))
        run["valid/loss"].append(np.mean(test_loss))
        run["train/within_5min_acc"].append(train_within_10min_correct / train_within_10min_total)
        run["test/within_5min_acc"].append(test_within_10min_correct / test_within_10min_total)
    
    run.stop()
    
    return model, final_loss


def main(argv):

    feature_names = ["time_limit_raw", "priority", "req_cpus", "req_mem", "req_nodes"]
    num_features = len(feature_names)
    num_jobs = 10000




    feature_names = ["time_limit_raw", "priority", "req_cpus", "req_mem", "req_nodes",
                     "jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue",
                     "nodes_ahead_queue", "time_limit_ahead_queue", "jobs_running",
                     "cpus_running", "memory_running", "nodes_running", "time_limit_running"
                     
                     ]
    num_features = len(feature_names)

    df = read_db.read_to_df(table="new_jobs_all", read_all=False, jobs=num_jobs)
    np_array = df.to_numpy()

    # Read in desired features and target columns to numpy arrays
    feature_indices = get_feature_indices(df, feature_names)
    target_index = get_planned_target_index(df)
    X, y = np_array[:, feature_indices], np_array[:, target_index]
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y = y / 60

    train_dataloader, test_dataloader = create_dataloaders(X, y)
    
    param_options = {
        # 'feature_names': [str(feature_names[0:5]),
        #                   str(feature_names)],
        'lr': [0.01, 0.001],
        'batch_size': [32, 128],
        'optimizer': ["adam", "sgd"],
        'hl1': [16, 32, 64],
        'hl2': [16, 32, 64],
    }
    
    epochs=20
    
    best_loss = 1e7
    best_params = {}
    
    
    keys = param_options.keys()
    combinations = list(itertools.product(*param_options.values()))
    for combo in combinations:
        combo_params = dict(zip(keys, combo))
        params = {
            'feature_names': str(feature_names),
            'num_features': num_features,
            'num_jobs': num_jobs,
            'lr': combo_params['lr'],
            'batch_size': combo_params['batch_size'],
            'epochs': epochs,
            'loss_fn': FLAGS.loss,
            'optimizer': combo_params['optimizer'],
            'hl1': combo_params['hl1'],
            'hl2': combo_params['hl2'],
            'dropout': FLAGS.dropout
        }
        model, loss = train(params, epochs, train_dataloader, test_dataloader)
        print(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params
    
    print(f"Best Loss: {best_loss}")
    print(f"Best Parameters: \n{str(best_params)}")

   #  plt.plot(train_loss_by_epoch)
   #  plt.plot(test_loss_by_epoch)
  #   plt.legend(['Train_loss', 'Test loss'])
  #  plt.show()

if __name__ == '__main__':
    app.run(main)
