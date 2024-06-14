# -*- coding: utf-8 -*-

from absl import app, flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import neptune
import neptune.integrations.optuna as npt_utils
import transformations
import optuna
from optuna.samplers import TPESampler
import time
import config_file

import classify_train

import read_db
from model import nn_model

FLAGS = flags.FLAGS

flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'Number of Epochs')
flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
flags.DEFINE_integer('hl1', 128, 'Hidden layer 1 dim')
flags.DEFINE_integer('hl2', 64, 'Hidden layer 1 dim')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate')
flags.DEFINE_enum('activ', 'relu', ['relu', 'leaky_relu', 'elu'], 'Activation function')
flags.DEFINE_boolean('transform', True,'Use transformations on features')
flags.DEFINE_boolean('shuffle', True,'Shuffle training/validation set')
flags.DEFINE_boolean('only_10min_plus', False, 'Only include jobs with planned longer than 10 minutes')
flags.DEFINE_boolean('transform_target', True, 'Whether or not to transform the planned variable')
flags.DEFINE_boolean('use_early_stopping', False, 'Whether or not to use early stopping')
flags.DEFINE_integer('early_stopping_patience', 10, 'Patience for early stopping')
flags.DEFINE_boolean('balance_dataset', False, 'Whether or not to use balance_dataset()')
flags.DEFINE_float('oversample', 0.4, 'Oversampling factor')
flags.DEFINE_float('undersample', 0.8, 'Undersampling factor')

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

def get_planned_target_index(df):
    return df.columns.get_loc('planned')

def get_feature_indices(df, feature_names):
    feature_indices = []
    for feature_name in feature_names:
        try:
            feature_indices.append(df.columns.get_loc(feature_name))
        except Exception as e:
            print(f"Error: Could not find '{feature_name}' in database\nExiting...")
            print(e)
            sys.exit(1)
    return feature_indices

def create_dataloaders(X, y):
    if FLAGS.balance_dataset:
        X, y = classify_train.balance_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9,
                                                        shuffle=FLAGS.shuffle,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9,
                                                      shuffle=FLAGS.shuffle,
                                                      random_state=42)
    if FLAGS.transform:
        _, X_test = transformations.scale_log(X_train, X_test)
        X_train, X_val = transformations.scale_log(X_train, X_val)

    if FLAGS.transform_target:
        _, y_test = transformations.scale_log(y_train, y_test)
        y_train, y_val = transformations.scale_log(y_train, y_val)

    x_train_to_tensor = torch.from_numpy(X_train).to(torch.float32)
    y_train_to_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
    y_test_to_tensor = torch.from_numpy(y_test).to(torch.float32)
    x_val_to_tensor = torch.from_numpy(X_val).to(torch.float32)
    y_val_to_tensor = torch.from_numpy(y_val).to(torch.float32)

    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    test_dataset = TensorDataset(x_test_to_tensor, y_test_to_tensor)
    val_dataset = TensorDataset(x_val_to_tensor, y_val_to_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size)
    return train_dataloader, test_dataloader, val_dataloader

def calculate_custom_loss(pred, y, train_test_or_val):
    sub_tensor = torch.sub(pred.flatten(), y)
    binary_within = torch.where(torch.abs(sub_tensor) < 10, 1, 0)
    pred[pred < 0] = 0
    custom_loss[f"{train_test_or_val}_within_10min_total"] += binary_within.shape[0]
    custom_loss[f"{train_test_or_val}_within_10min_correct"] += binary_within.sum().item()

    y_binary = torch.where(y > 10, 1, 0)
    pred_binary = torch.where(pred.flatten() > 10, 1, 0)
    binary_10min = torch.where(y_binary == pred_binary, 1, 0)
    custom_loss[f"{train_test_or_val}_binary_10min_total"] += binary_10min.shape[0]
    custom_loss[f"{train_test_or_val}_binary_10min_correct"] += binary_10min.sum().item()

def objective(trial):
    global custom_loss
    
    starting_time = time.time()

    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem",
                     "jobs_ahead_queue", "jobs_running", "cpus_ahead_queue",
                     "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
                     "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    num_features = len(feature_names)
    num_jobs = 400_000

    params = {
        'feature_names': str(feature_names),
        'num_features': num_features,
        'num_jobs': num_jobs,
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_int('batch_size', 16, 128),
        'epochs': FLAGS.epochs,
        'loss_fn': trial.suggest_categorical('loss_fn', ['mse_loss', 'l1_loss', 'smooth_l1_loss']),
        'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw']),
        'hl1': trial.suggest_int('hl1', 32, 256),
        'hl2': trial.suggest_int('hl2', 16, 128),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'activ': trial.suggest_categorical('activ', ['relu', 'leaky_relu', 'elu']),
    }

    num_features = len(feature_names)

    model = nn_model(num_features, params['hl1'], params['hl2'], params['dropout'], params['activ'])

    if params['loss_fn'] == "l1_loss":
        loss_fn = nn.L1Loss()
    elif params['loss_fn'] == "mse_loss":
        loss_fn = nn.MSELoss()
    elif params['loss_fn'] == "smooth_l1_loss":
        loss_fn = nn.SmoothL1Loss()
    else:
        sys.exit(f"Loss function '{params['loss_fn']}' not supported")

    if params['optimizer'] == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=params['lr'])
    elif params['optimizer'] == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=params['lr'])
    elif params['optimizer'] == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=params['lr'])
    else:
        sys.exit(f"Optimizer '{params['optimizer']}' not supported")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['epochs']):
        train_loss = []
        val_loss = []
        custom_loss = dict.fromkeys(custom_loss, 0)

        model.train()
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            calculate_custom_loss(pred.flatten(), y, "train")

        model.eval()
        for X, y in val_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                val_loss.append(loss.item())
                calculate_custom_loss(pred.flatten(), y, "val")

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= FLAGS.early_stopping_patience and FLAGS.use_early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print(f"Epoch = {epoch}, Train_loss = {avg_train_loss:.2f}, Test Loss = {avg_val_loss:.5f}")

    ending_time = time.time()
    
    print(f"Objective took {ending_time - starting_time} seconds")

    return avg_val_loss

def detailed_objective(trial):
    global custom_loss
    
    starting_time = time.time()

    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem",
                     "jobs_ahead_queue", "jobs_running", "cpus_ahead_queue",
                     "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
                     "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    num_features = len(feature_names)
    num_jobs = 400_000

    params = {
        'feature_names': str(feature_names),
        'num_features': num_features,
        'num_jobs': num_jobs,
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_int('batch_size', 16, 128),
        'epochs': FLAGS.epochs,
        'loss_fn': trial.suggest_categorical('loss_fn', ['mse_loss', 'l1_loss', 'smooth_l1_loss']),
        'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw']),
        'hl1': trial.suggest_int('hl1', 32, 256),
        'hl2': trial.suggest_int('hl2', 16, 128),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'activ': trial.suggest_categorical('activ', ['relu', 'leaky_relu', 'elu']),
    }

    num_features = len(feature_names)
    
    

    model = nn_model(num_features, params['hl1'], params['hl2'], params['dropout'], params['activ'])

    if params['loss_fn'] == "l1_loss":
        loss_fn = nn.L1Loss()
    elif params['loss_fn'] == "mse_loss":
        loss_fn = nn.MSELoss()
    elif params['loss_fn'] == "smooth_l1_loss":
        loss_fn = nn.SmoothL1Loss()
    else:
        sys.exit(f"Loss function '{params['loss_fn']}' not supported")

    if params['optimizer'] == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=params['lr'])
    elif params['optimizer'] == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=params['lr'])
    elif params['optimizer'] == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=params['lr'])
    else:
        sys.exit(f"Optimizer '{params['optimizer']}' not supported")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['epochs']):
        train_loss = []
        val_loss = []
        custom_loss = dict.fromkeys(custom_loss, 0)

        model.train()
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            calculate_custom_loss(pred.flatten(), y, "train")

        model.eval()
        for X, y in val_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                val_loss.append(loss.item())
                calculate_custom_loss(pred.flatten(), y, "val")

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= FLAGS.early_stopping_patience and FLAGS.use_early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print(f"Epoch = {epoch}, Train_loss = {avg_train_loss:.2f}, Test Loss = {avg_val_loss:.5f}")

    ending_time = time.time()
    
    print(f"Objective took {ending_time - starting_time} seconds")
    
    
    model.eval()
    test_loss = []
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            test_loss.append(loss.item())
            calculate_custom_loss(pred.flatten(), y, "test")
    avg_test_loss = np.mean(test_loss)
    
    return avg_val_loss, avg_test_loss

def main(argv):
    run = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
        tags=["regression"]
    )
    
    neptune_callback = npt_utils.NeptuneCallback(run)
    
    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem",
                     "jobs_ahead_queue", "jobs_running", "cpus_ahead_queue",
                     "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
                     "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    num_jobs = 400_000
    read_all = True if num_jobs == 0 else False
    
    df = read_db.read_to_df(table="new_jobs_all", read_all=read_all, jobs=num_jobs)
    if FLAGS.only_10min_plus:
        df = df[df['planned'] > 10 * 60]
    np_array = df.to_numpy()
    
    feature_indices = get_feature_indices(df, feature_names)
    target_index = get_planned_target_index(df)
    X, y = np_array[:, feature_indices], np_array[:, target_index]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    y = y / 60

    global train_dataloader
    global test_dataloader
    global val_dataloader
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X, y)
    
    
    sampler = TPESampler(n_startup_trials=10)                      
    
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
        
    results = detailed_objective(best_trial)
    print(results)
    
    run.stop()
        
    

if __name__ == '__main__':
    app.run(main)
