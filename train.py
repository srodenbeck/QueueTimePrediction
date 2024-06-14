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
import smogn
from scipy.stats import pearsonr


import classify_train

import config_file
import read_db
from model import nn_model



flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of Epochs')
flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
flags.DEFINE_integer('hl1', 128, 'Hidden layer 1 dim')
flags.DEFINE_integer('hl2', 64, 'Hidden layer 1 dim')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate')
flags.DEFINE_boolean('transform', True,'Use transformations on features')
flags.DEFINE_enum('activ', 'relu', ['relu', 'leaky_relu', 'elu'], 'Activation function')
flags.DEFINE_boolean('shuffle', True,'Shuffle training/validation set')
flags.DEFINE_boolean('only_10min_plus', True, 'Only include jobs with planned longer than 10 mintues')
flags.DEFINE_boolean('transform_target', False, 'Whether or not to transform the planned variable')
flags.DEFINE_boolean('use_early_stopping', True, 'Whether or not to use early stopping')
flags.DEFINE_integer('early_stopping_patience', 10, 'Patience for early stopping')

flags.DEFINE_boolean('balance_dataset', False, 'Whether or not to use balance_dataset()')

flags.DEFINE_float('oversample', 0.4, 'Oversampling factor')
flags.DEFINE_float('undersample', 0.8, 'Undersampling factor')


FLAGS = flags.FLAGS

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

def create_dataloaders(X, y):
    """
    create_dataloaders()
    
    Creates pytorch dataloaders from the given information

    Parameters
    ----------
    X : np array
        2-D Array of input features.
    y : np array
        Array of target feature.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for training data.
    test_dataloader : torch.utils.data.DataLoader
        Dataloader for testing data.

    """
    # EXPERIMENTAL
    if FLAGS.balance_dataset:
        X, y = classify_train.balance_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9,
                                                        shuffle=FLAGS.shuffle,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9,
                                                      shuffle=FLAGS.shuffle,
                                                      random_state=42)
    
    if FLAGS.transform:
        # X_train, X_test = transformations.scale_min_max(X_train, X_test)
        _, X_test = transformations.scale_log(X_train, X_test)
        X_train, X_val = transformations.scale_log(X_train, X_val)

    if FLAGS.transform_target:
        _, y_test = transformations.scale_log(y_train, y_test)
        y_train, y_val = transformations.scale_log(y_train, y_val)

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
    x_val_to_tensor = torch.from_numpy(X_val).to(torch.float32)
    y_val_to_tensor = torch.from_numpy(y_val).to(torch.float32)

    # Second step: Creating TensorDataset for Dataloader
    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    test_dataset = TensorDataset(x_test_to_tensor, y_test_to_tensor)
    val_dataset = TensorDataset(x_val_to_tensor, y_val_to_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size)
    
    return train_dataloader, test_dataloader, val_dataloader


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

def main(argv):
    global custom_loss
    
    # Connect to neptune
    run = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
        tags=["regression"]
    )

    # Feature names to use in training
    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem",
                     "jobs_ahead_queue", "jobs_running", "cpus_ahead_queue",
                     "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
                     "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    num_features = len(feature_names)
    num_jobs = 4_000_000
    read_all = True if num_jobs == 0 else False

    # Specified parameters to upload to neptune
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
        'dropout': FLAGS.dropout,
        '10_min_plus': FLAGS.only_10min_plus
    }
    run["parameters"] = params

    num_features = len(feature_names)
    print("Reading from database")
    df = read_db.read_to_df(table="new_jobs_all", read_all=read_all, jobs=num_jobs)
    print("Finished reading database")
    
    # Transform data if only_10min_plus flag is on, discards data with planned
    # less than 10 minutes
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

    # Make dataloaders from arrays
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X, y)

    # Create model
    model = nn_model(num_features, FLAGS.hl1, FLAGS.hl2, FLAGS.dropout, FLAGS.activ)

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
    val_loss_by_epoch = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(FLAGS.epochs):
        train_loss = []
        val_loss = []
        custom_loss = dict.fromkeys(custom_loss, 0)

        # Training
        model.train()
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            calculate_custom_loss(pred.flatten(), y, "train")

        # Evaluation/Validation
        model.eval()
        for X, y in val_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                val_loss.append(loss.item())
                calculate_custom_loss(pred.flatten(), y, "val")
                if epoch == FLAGS.epochs - 1:
                    for i in range(y.shape[0]):
                        print(f"Predicted: {pred.flatten()[i]} -- Real: {y[i]}")

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= FLAGS.early_stopping_patience and FLAGS.use_early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            for i in range(y.shape[0]):
                print(f"Predicted: {pred.flatten()[i]} -- Real: {y[i]}")
            break

        print(f"Epoch = {epoch}, Train_loss = {avg_train_loss:.2f}, Validation Loss = {avg_val_loss:.5f}")
        train_loss_by_epoch.append(avg_train_loss)
        val_loss_by_epoch.append(avg_val_loss)
        run["train/loss"].append(avg_train_loss)
        run["valid/loss"].append(avg_val_loss)
        run["train/within_10min_acc"].append(custom_loss["train_within_10min_correct"] / custom_loss["train_within_10min_total"] * 100)
        run["valid/within_10min_acc"].append(custom_loss["val_within_10min_correct"] / custom_loss["val_within_10min_total"] * 100)
        run["train/binary_10min_acc"].append(custom_loss["train_binary_10min_correct"] / custom_loss["train_binary_10min_total"] * 100)
        run["valid/binary_10min_acc"].append(custom_loss["val_binary_10min_correct"] / custom_loss["val_binary_10min_total"] * 100)

    # Graphing and getting R2 value of model pred vs actual
    print("Graphing pred vs actual and calculating pearsons r")
    torch.eval()
    y_pred = []
    y_actual= []
    test_loss = []
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            test_loss.append(loss.item())
            calculate_custom_loss(pred.flatten(), y, "test")
            
            y_pred.extend(pred.flatten())
            y_actual.extend(y)
    avg_test_loss = np.mean(np.mean(test_loss))
    print(f"Average test loss of {avg_test_loss}")
    
    r_value = pearsonr(y_pred, y_actual)
    print("pearson's r: ", r_value)
    plt.scatter(y_pred, y_actual)
    plt.xlabel("y predicted")
    plt.ylabel("y")
    plt.show()

    
    

    run.stop()

if __name__ == '__main__':
    app.run(main)
