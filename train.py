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
import transformations
# import smogn

import config_file
import read_db
from model import nn_model


flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('lr', 0.01, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of Epochs')
flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
flags.DEFINE_integer('hl1', 32, 'Hidden layer 1 dim')
flags.DEFINE_integer('hl2', 16, 'Hidden layer 1 dim')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate')
flags.DEFINE_boolean('transform', False,'Use transformations on features')
flags.DEFINE_boolean('shuffle', True,'Shuffle training/validation set')

FLAGS = flags.FLAGS

custom_loss = {
    "train_within_10min_correct":  0,
    "train_within_10min_total":  0,
    "test_within_10min_correct":  0,
    "test_within_10min_total":  0,
    "train_binary_10min_correct":  0,
    "train_binary_10min_total":  0,
    "test_binary_10min_correct":  0,
    "test_binary_10min_total":  0
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
            sys.exit(1)
    return feature_indices


def create_dataloaders(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        shuffle=FLAGS.shuffle,
                                                        random_state=0)
    if FLAGS.transform:
        X_train, X_test = transformations.scale_min_max(X_train, X_test)

    # for i in range(X_train.shape[1]):
        # print(X_train[:, i])
        # print(min(X_train[:, i]))
        # X_train[:, i] = np.log(X_train[:, i])
    #     # X_train[:, i], X_test[:, i], lmbda = transformations.boxcox(X_train[:, i],
    #     #                                                             X_test[:, i])
    #     # print(f"Lambda of {lmbda}")

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


def calculate_custom_loss(pred, y, train_or_test):
    sub_tensor = torch.sub(pred.flatten(), y)
    binary_within = torch.where(torch.abs(sub_tensor) < 10, 1, 0)
    pred[pred < 0] = 0
    custom_loss[f"{train_or_test}_within_10min_total"] += binary_within.shape[0]
    custom_loss[f"{train_or_test}_within_10min_correct"] += binary_within.sum().item()

    y_binary = torch.where(y > 10, 1, 0)
    pred_binary = torch.where(pred.flatten() > 10, 1, 0)
    binary_10min = torch.where(y_binary == pred_binary, 1, 0)
    custom_loss[f"{train_or_test}_binary_10min_total"] += binary_10min.shape[0]
    custom_loss[f"{train_or_test}_binary_10min_correct"] += binary_10min.sum().item()



def main(argv):
    global custom_loss
    run = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
    )

    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_nodes", "req_mem", "jobs_ahead_queue", "jobs_running"]
    num_features = len(feature_names)
    num_jobs = 100000
    read_all = True if num_jobs == 0 else False

    params = {
        'feature_names': str(feature_names),
        'num_features': num_features,
        'num_jobs': num_jobs,
        'lr': FLAGS.lr,
        'batch_size': FLAGS.batch_size,
        'epochs': FLAGS.epochs,
        'loss_fn': FLAGS.loss,
        'optimizer': FLAGS.optimizer,
        'hl1': FLAGS.hl1,
        'hl2': FLAGS.hl2,
        'dropout': FLAGS.dropout
    }
    run["parameters"] = params

    num_features = len(feature_names)
    df = read_db.read_to_df(table="new_jobs_all", read_all=read_all, jobs=num_jobs)
    np_array = df.to_numpy()

    # Read in desired features and target columns to numpy arrays
    feature_indices = get_feature_indices(df, feature_names)
    target_index = get_planned_target_index(df)
    X, y = np_array[:, feature_indices], np_array[:, target_index]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Transformations
    y = y / 60

    train_dataloader, test_dataloader = create_dataloaders(X, y)

    model = nn_model(num_features, FLAGS.hl1, FLAGS.hl2, FLAGS.dropout)

    # loss function
    if FLAGS.loss == "l1_loss":
        loss_fn = nn.L1Loss
    elif FLAGS.loss == "mse_loss":
        loss_fn = nn.MSELoss()
    elif FLAGS.loss == "smooth_l1_loss":
        loss_fn = nn.SmoothL1Loss()
    else:
        sys.exit(f"Loss function '{FLAGS.loss}' not supported")

    # Optimizer
    if FLAGS.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=FLAGS.lr)
    else:
        sys.exit(f"Optimizer '{FLAGS.optimizer}' not supported")

    # Run training loop
    train_loss_by_epoch = []
    test_loss_by_epoch = []

    for epoch in range(FLAGS.epochs):
        train_loss = []
        test_loss = []
        custom_loss = dict.fromkeys(custom_loss, 0)
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            calculate_custom_loss(pred.flatten(), y, "train")

        for X, y in test_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                test_loss.append(loss.item())
                calculate_custom_loss(pred.flatten(), y, "test")
                if epoch == FLAGS.epochs - 1:
                    for i in range(y.shape[0]):
                        print(f"Predicted: {pred.flatten()[i]} -- Real: {y[i]}")

        print(f"Epoch = {epoch}, Train_loss = {np.mean(train_loss):.2f}, Test Loss = {np.mean(test_loss):.5f}")
        train_loss_by_epoch.append(np.mean(train_loss))
        test_loss_by_epoch.append(np.mean(test_loss))
        run["train/loss"].append(np.mean(train_loss))
        run["valid/loss"].append(np.mean(test_loss))
        run["train/within_10min_acc"].append(custom_loss["train_within_10min_correct"] / custom_loss["train_within_10min_total"] * 100)
        run["valid/within_10min_acc"].append(custom_loss["test_within_10min_correct"] / custom_loss["test_within_10min_total"] * 100)
        run["train/binary_10min_acc"].append(custom_loss["train_binary_10min_correct"] / custom_loss["train_binary_10min_total"] * 100)
        run["valid/binary_10min_acc"].append(custom_loss["test_binary_10min_correct"] / custom_loss["test_binary_10min_total"] * 100)

    run.stop()

if __name__ == '__main__':
    app.run(main)
