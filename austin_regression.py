from absl import app, flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import neptune
import transformations
import smogn
from scipy.stats import pearsonr
import transformations
import pandas as pd

import classify_train

import config_file
import read_db
from model import nn_model



# flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
# flags.DEFINE_float('lr', 0.001, 'Learning rate.')
# flags.DEFINE_integer('batch_size', 128, 'Batch size')
# flags.DEFINE_integer('epochs', 50, 'Number of Epochs')
# flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
# flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
# flags.DEFINE_integer('hl1', 32, 'Hidden layer 1 dim')
# flags.DEFINE_integer('hl2', 78, 'Hidden layer 1 dim')
# flags.DEFINE_float('dropout', 0.15, 'Dropout rate')
# flags.DEFINE_boolean('transform', True,'Use transformations on features')
# flags.DEFINE_enum('activ', 'leaky_relu', ['relu', 'leaky_relu', 'elu', 'gelu'], 'Activation function')
# flags.DEFINE_boolean('shuffle', False,'Shuffle training/validation set')
# flags.DEFINE_boolean('only_10min_plus', True, 'Only include jobs with planned longer than 10 mintues')
# flags.DEFINE_boolean('use_early_stopping', False, 'Whether or not to use early stopping')
# flags.DEFINE_integer('early_stopping_patience', 10, 'Patience for early stopping')
# flags.DEFINE_boolean('condense_same_times', False, 'Whether or not to remove jobs submitted back to back, bar the first job')

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

def create_timeseries_folds(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv

def create_dataloaders_tscv(X, y, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    if FLAGS.transform:
        # X_train, X_test = transformations.scale_min_max(X_train, X_test)
        X_train, X_test = transformations.scale_log(X_train, X_test)

    x_train_to_tensor = torch.from_numpy(X_train).to(torch.float32)
    y_train_to_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
    y_test_to_tensor = torch.from_numpy(y_test).to(torch.float32)
    
    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    test_dataset = TensorDataset(x_test_to_tensor, y_test_to_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size)
    
    return train_dataloader, test_dataloader
    

def create_dataloaders(X, y):
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
    # Calculating loss in regards to how many target values were within 10
    # minutes of predicted values
    sub_tensor = torch.sub(pred.flatten(), y)
    binary_within = torch.where(torch.abs(sub_tensor) < 60, 1, 0)
    pred[pred < 0] = 0
    custom_loss[f"{train_test_val}_within_1hr_total"] += binary_within.shape[0]
    custom_loss[f"{train_test_val}_within_1hr_correct"] += binary_within.sum().item()

def load_data(read_all=True, num_jobs=0, feature_names=None):
    print("Reading from database")
    df = read_db.read_to_df(table="jobs_everything_tmp", read_all=read_all, jobs=num_jobs, order_by="eligible",
                            condense_same_times=FLAGS.condense_same_times)
    df['eligible'] = pd.to_datetime(df['eligible'])
    print("Finished reading database")
    print("DataFrame has shape", df.shape)

    transformed_cols = []
    if "partition" in feature_names:
        df = transformations.make_one_hot(df, "partition")
        transformed_cols.append("partition")

    features = []
    for col_name in df.columns:
        if col_name in transformed_cols:
            continue
        for name in feature_names:
            if name in col_name:
                features.append(col_name)
                break
    feature_names = features
    num_features = len(feature_names)
    print(f"{feature_names=}")
    print(f"{num_features=}")

    # Transform data if only_10min_plus flag is on, discards data with planned < 10 minutes.
    df = df[df['planned'] > 10 * 60]
    print(f"Using {len(df)} jobs")

    np_array = df.to_numpy()

    # Read in desired features and target columns to numpy arrays
    feature_indices = get_feature_indices(df, feature_names)
    target_index = get_planned_target_index(df)
    X_rows, y_rows = np_array[:, feature_indices], np_array[:, target_index]
    X_rows = X_rows.astype(np.float32)
    y_rows = y_rows.astype(np.float32)
    print("X shape:", X_rows.shape)
    print("y shape:", y_rows.shape)
    return X_rows, y_rows, np_array

def start_trials():
    global custom_loss
    
    # Connect to neptune
    run = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
        tags=["austin_regression"]
    )

    # Feature names to use in training
    feature_names = classify_train.feature_options("austin_hypo_2")
    num_features = len(feature_names)
    num_jobs = 0
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

    X_rows, y_rows, np_array = load_data(read_all, num_jobs, feature_names)


    # Transformations
    y_rows = y_rows / 60
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(np_array) // (2 * n_splits + 1))
    
    loss_by_fold = []
    for train_index, test_index in tscv.split(np_array):
        # train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X, y)
        train_dataloader, test_dataloader = create_dataloaders_tscv(X_rows, y_rows, train_index, test_index)
    
        # Create model
        model = nn_model(num_features, FLAGS.hl1, FLAGS.hl2, FLAGS.dropout, FLAGS.activ)
        # loss_fn = nn.L1Loss
        loss_fn = nn.SmoothL1Loss()
        optimizer = optim.Adam(params=model.parameters(), lr=FLAGS.lr)

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
            # test_dataloader with tscv, val_dataloader otherwise
            for X, y in test_dataloader:
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
        model.eval()
        y_pred = []
        y_actual= []
        test_loss = []
        absolute_percentage_error = []
        with torch.no_grad():
            for X, y in test_dataloader:
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
        
        loss_by_fold.append(avg_test_loss)
        
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

    
    print(loss_by_fold)
    print(np.mean(loss_by_fold))
    
    # Save models state dict
    torch.save(model.state_dict(), "regr_model.pt")
    
    
    run.stop()

def main(argv):
    start_trials()

if __name__ == '__main__':
    app.run(main)
