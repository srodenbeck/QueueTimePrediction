from absl import app, flags
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
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
import seaborn as sns

import classify_train

import config_file
import read_db
from model import nn_model
import shap


# 493.01347732543945 %



flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('epochs', 50, 'Number of Epochs')
flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
flags.DEFINE_integer('hl1', 32, 'Hidden layer 1 dim')
flags.DEFINE_integer('hl2', 64, 'Hidden layer 2 dim')
flags.DEFINE_integer('hl3', 32, 'Hidden layer 3 dim')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate')
flags.DEFINE_boolean('transform', True,'Use transformations on features')
flags.DEFINE_enum('activ', 'elu', ['relu', 'leaky_relu', 'elu', 'gelu'], 'Activation function')
flags.DEFINE_boolean('shuffle', False,'Shuffle training/validation set')
flags.DEFINE_boolean('only_10min_plus', True, 'Only include jobs with planned longer than 10 mintues')
flags.DEFINE_boolean('transform_target', False, 'Whether or not to transform the planned variable')
flags.DEFINE_boolean('use_early_stopping', False, 'Whether or not to use early stopping')
flags.DEFINE_integer('early_stopping_patience', 10, 'Patience for early stopping')

flags.DEFINE_boolean('balance_dataset', False, 'Whether or not to use balance_dataset()')
flags.DEFINE_boolean('condense_same_times', False, 'Whether or not to remove jobs submitted back to back, bar the first job')

flags.DEFINE_float('oversample', 0.4, 'Oversampling factor')
flags.DEFINE_float('undersample', 0.8, 'Undersampling factor')

flags.DEFINE_boolean('ten_thousand_or_below', False, 'Whether or not to limit jobs to jobs with planned times of 10_000 minutes or below')

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


# Format is - partition: [#Nodes, #CPU_cores, Cores/Node, Mem/Node (GB)]
#TODO: Add number of gpus
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

def create_timeseries_folds(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv

def create_dataloaders_tscv(X, y, train_index, test_index):
    """
    create_dataloaders_tscv

    Parameters
    ----------
    X : np array
        2-D Array of input features.
    y : np array
        2-D Array of target features.
    train_index : int array
        int array of input indices.
    test_index : int array
        int array of target indices.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for training data.
    test_dataloader : torch.utils.data.DataLoader
        Dataloader for testing data.

    """
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # TODO: Shuffle X_train and y_train
    
    
    if FLAGS.transform:
        X_train, X_test = transformations.scale_min_max(X_train, X_test)
        # X_train, X_test = transformations.scale_log(X_train, X_test, 0)

    if FLAGS.transform_target:
        y_train, y_test = transformations.scale_log(y_train, y_test)
        
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
    
    
    # All features
    # ["priority", "time_limit_raw", "req_cpus", "req_mem",
    #                  "jobs_ahead_queue", "jobs_running", "cpus_ahead_queue",
    #                  "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
    #                  "cpus_running", "memory_running", "nodes_running", "time_limit_running",
    #                  "year", "month", "day", "hour", "minute", "day_of_week", "day_of_year",
    #                  "par_jobs_ahead_queue", "par_cpu_ahead_queue", "part_memory_ahead_queue",
    #                  "par_nodes_ahead_queue", "par_time_ahead_queue", "par_jobs_running",
    #                  "par_cpus_running", "par_memory_running", "par_nodes_running", "par_time_limit_running"]
    
    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
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
                     # "partition", "qos"]
    num_features = len(feature_names)
    print(num_features)
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
        'hl3': FLAGS.hl3,
        'dropout': FLAGS.dropout,
        '10_min_plus': FLAGS.only_10min_plus
    }
    run["parameters"] = params

    num_features = len(feature_names)
    print("Reading from database")
    df = read_db.read_to_df(table="jobs_everything_all_2", read_all=read_all, jobs=num_jobs, condense_same_times=False)
            
    df = df[df['partition'] != 'gpu']
    print("Finished reading from database")
    
    temp_df = df['partition'].map(partition_feature_dict).apply(pd.Series)
    temp_df = temp_df.fillna(1)
        
    # Rename the columns in the temporary dataframe
    temp_df.columns = ['par_total_nodes', 'par_total_cpu', 'par_cpu_per_node', 'par_mem_per_node', 'par_total_gpu']
    
    # Concatenate the original dataframe with the temporary dataframe
    df = pd.concat([df, temp_df], axis=1)
    df = df.fillna(1)
    
    if FLAGS.condense_same_times:
        prev_user = None
        rows_to_drop = []
        for index, row in df.iterrows():
            if prev_user == row['user']:
                rows_to_drop.append(index)
            prev_user = row['user']
    
        df = df.drop(rows_to_drop[1::2])
    
    
    df['eligible'] = pd.to_datetime(df['eligible'])
    df['year'] = df['eligible'].dt.year
    df['month'] = df['eligible'].dt.month
    df['day'] = df['eligible'].dt.day
    df['hour'] = df['eligible'].dt.hour
    df['minute'] = df['eligible'].dt.minute
    df['day_of_week'] = df['eligible'].dt.dayofweek
    df['day_of_year'] = df['eligible'].dt.dayofyear
    
    
    print("Finished reading database")
    print("DataFrame has shape", df.shape)
    
     
    
    # Removing values over 75000
    if True:
        print("Removing values over 75_000 minutes")
        df = df[df['planned'] < 75_000 * 60]
        
        
    # Average test loss of 24512.13562685602
    # Mean absolute percentage error: 169.36719417572021 %
    # pearson's r:  PearsonRResult(statistic=0.2133563214593553, pvalue=2.119680744947272e-89)
    
    transformed_cols = []
    # Feature manipulation for categorical features if needed
    if "account" in feature_names:
        df['account'] = df['account'].apply(transformations.accountToNormUsage)
        transformed_cols.append("account")
    if "partition" in feature_names: 
        df = transformations.make_one_hot(df, "partition")
        transformed_cols.append("partition")
    if "qos" in feature_names:
        df = transformations.make_one_hot(df, "qos", new_col_limit=4)
        transformed_cols.append("qos")
    
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
    print(feature_names)
    print(num_features)



    if FLAGS.ten_thousand_or_below:
        df = df[df['planned'] < 10_000 * 60]
        
        
        
    # Transform data if only_10min_plus flag is on, discards data with planned
    # less than 10 minutes
    if FLAGS.only_10min_plus:
        print("10 plus flag: ", FLAGS.only_10min_plus)
        # TODO: change back to 10 min
        df = df[df['planned'] > 10 * 60]
        # Limit to 100_000 jobs for better comparison
        df = df.iloc[:1_000_000]
        # df['planned'] += 1
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

    new_df = pd.DataFrame(data=X_rows, columns=feature_names)
    # new_df['target'] = y_rows
    
    print("here")
    
    # Transformations
    y_rows = y_rows / 60
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)   # , test_size = len(np_array) // (2 * n_splits + 1))
    
    loss_by_fold = []
    print("Loading sklearn ")
    from sklearn.ensemble import RandomForestRegressor
    print("getting importances")
    # importances = RandomForestRegressor().fit(X_rows, y_rows).feature_importances_
    # important_indices = np.argsort(importances)[-10:]
    # print(important_indices)

    test_rows = None
    best_loss = float('inf')
    best_model = None

    model_idx = 0
    total_models = 0
    
    for train_index, test_index in tscv.split(np_array):
        total_models += 1
        test_rows = new_df.iloc[test_index]
        
        # Make dataloaders from arrays
        # train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X, y)
        train_dataloader, test_dataloader = create_dataloaders_tscv(X_rows, y_rows, train_index, test_index)
    
        # Create model
        model = nn_model(num_features, FLAGS.hl1, FLAGS.hl2, FLAGS.hl3, FLAGS.dropout, FLAGS.activ)
    
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
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                loss.backward()
                nn_utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
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
        X_test = []
        absolute_percentage_error = []
        last_dataloader = test_dataloader
        within_50_percent = 0.0
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                test_loss.append(loss.item())
                X_test.extend(X)
                flat_pred = pred.flatten()
                for idx in range(len(y)):
                    ape = abs((y[idx] - flat_pred[idx]) / y[idx])
                    if flat_pred[idx] * 0.5 < y[idx] < flat_pred[idx] * 1.5:
                        within_50_percent += 1
                    absolute_percentage_error.append(ape)
                    if ape < 0.1 or ape > 10:
                        pass
                        
                
                calculate_custom_loss(pred.flatten(), y, "test")
                y_pred.extend(pred.flatten())
                y_actual.extend(y)
                
        within_50_percent /= len(absolute_percentage_error)
        print("Percent of jobs within 50% of predicted: ", within_50_percent * 100)
                
        test_rows = (test_rows-test_rows.min())/(test_rows.max()-test_rows.min())
        
        test_rows['y_pred'] = y_pred
        test_rows['y_actual'] = y_actual
        test_rows['error'] = absolute_percentage_error
        
        # Looking for patterns in low accuracy vs high accuracy predictions
        low_threshold = 0.1
        high_threshold = 10.0
        
        
        test_rows['group'] = 'other'
        test_rows.loc[test_rows['error'] <= low_threshold, 'group'] = 'close'
        test_rows.loc[test_rows['error'] > high_threshold, 'group'] = 'far_off'
        
        # Separate the groups
        close_predictions = test_rows[test_rows['group'] == 'close']
        far_off_predictions = test_rows[test_rows['group'] == 'far_off']
        
        # Drop columns not needed for analysis
        close_features = close_predictions.drop(columns=['group'])
        far_off_features = far_off_predictions.drop(columns=['group'])

        
        # Box Plots
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data=close_features)
        plt.title('Close Predictions Features')
        plt.ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()
        
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data=far_off_features)
        plt.title('Far-off Predictions Features')
        plt.ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()
        
        # # Histograms
        # xlims = []
        # for feat in close_features.columns:
        #     lower = min(min(close_features[feat]), min(far_off_features[feat]))
        #     upper = max(max(close_features[feat]), max(far_off_features[feat]))
        #     if lower == upper:
        #         upper += 1
        #     xlims.append((lower, upper))
            
        
        # axes = close_features.hist(figsize=(12, 12))
        # axes = axes.flatten()
        # for i, ax in enumerate(axes):
        #     ax.set_xlim(xlims[i])
        # plt.suptitle('Close Predictions Features Distribution')
        # plt.show()
        
        
        # axes = far_off_features.hist(figsize=(12, 12))
        # axes = axes.flatten()
        # for i, ax in enumerate(axes):
        #     ax.set_xlim(xlims[i])
        # plt.suptitle('Far-off Predictions Features Distribution')
        # plt.show()
        
        
        
        # Correlation Heatmaps
        plt.figure(figsize=(10, 8))
        sns.heatmap(close_features.corr(), annot=True, cmap='coolwarm')
        plt.title('Close Predictions Features Correlation')
        plt.show()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(far_off_features.corr(), annot=True, cmap='coolwarm')
        plt.title('Far-off Predictions Features Correlation')
        plt.show()

        model.eval()
        
        def predict_function(data):
            data_tensor = torch.tensor(data, dtype=torch.float32)
            with torch.no_grad():
                return model(data_tensor).numpy().flatten()
        
        background_data = []
        for data in train_dataloader:
            inputs, _ = data
            background_data.append(inputs.numpy())
            if len(background_data) >= 100:
                break
            
            
        background_data = np.concatenate(background_data, axis=0)[:50]
        
        test_data = []
        for data in test_dataloader:
            inputs, _ = data
            test_data.append(inputs.numpy())
        test_data = np.concatenate(test_data, axis=0)
        
        X_to_explain = test_data[50:55]
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(predict_function, background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_to_explain)
        
        # Visualize SHAP values for the first instance in the test set
        shap.initjs()
        shap.summary_plot(shap_values, X_to_explain, feature_names=feature_names)
        # shap.force_plot(explainer.expected_value, shap_values[0], test_data[0])
        
        
        avg_test_loss = np.mean(np.mean(test_loss))
        
        if avg_test_loss < best_loss and total_models != n_splits:
            best_model = model
            model_idx = total_models
        
        loss_by_fold.append(np.mean(absolute_percentage_error))
        
        y_pred = np.array(y_pred)
        y_actual = np.array(y_actual)
        
        print(f"Average test loss of {avg_test_loss}")
        bin_width = 0.25
        bins = np.arange(0, 5 + bin_width, bin_width)
        plt.hist(absolute_percentage_error, bins=bins, density=True)
        plt.xlabel('Absolute Percentage Error')
        plt.ylabel('Probability Density')
        plt.title("Density Histogram of Absolute Percentage Error of Test Data")
        plt.xlim(0, 10)
        plt.show()
        
        
        
        # Average absolute percentage error
        avg_ape = np.mean(absolute_percentage_error)
        print("Mean absolute percentage error:", avg_ape * 100, "%")
        
        r_value = pearsonr(y_pred, y_actual)
        print("pearson's r: ", r_value)
        
        differences = y_pred - y_actual
        plt.hist(differences, 30)
        plt.xlabel("Difference between y pred and y actual")
        plt.ylabel("Count")
        plt.title("Histogram of Difference between y pred vs y actual")
        plt.xlim(-5000, 5000)
        plt.show()
        
        plt.scatter(y_pred, y_actual)
        plt.title("Scatter plot of y pred vs y actual with LOBF")
        plt.xlabel("y predicted")
        plt.ylabel("y actual")
        plt.xlim(0, 10_000)
        plt.ylim(0, 10_000)
        m, b = np.polyfit(y_pred, y_actual, 1)
        plt.plot(y_pred, m * y_pred + b, 'r-', label=f"Line of Best Fit (r={r_value})")
        plt.show()

        total = 0
        within_100_perc = 0
        within_200_perc = 0
        within_100_perc_plus1hr_buffer = 0
        within_50_perc_plus1hr_buffer = 0
        my_metric_1 = 0

        for z in range(len(y_pred)):
            total += 1
            if y_pred[z] > y_actual[z]:
                max_val = y_pred[z]
                min_val = y_actual[z]
            else:
                min_val = y_pred[z]
                max_val = y_actual[z]
            if (((max_val - min_val) / min_val) * 100) < 200:
                within_200_perc += 1
                if (((max_val - min_val) / min_val) * 100) < 100:
                    within_100_perc += 1

            if y_pred[z] < 1 * 60:
                if max_val - min_val < 20:
                    my_metric_1 += 1
            elif y_pred[z] < 5 * 60:
                if max_val - min_val < 60:
                    my_metric_1 += 1
            elif y_pred[z] < 12 * 60:
                if max_val - min_val < 120:
                    my_metric_1 += 1
            else:
                if max_val - min_val < 12 * 60:
                    my_metric_1 += 1

            min_val += 60
            max_val += 60
            if (((max_val - min_val) / min_val) * 100) < 100:
                within_100_perc_plus1hr_buffer += 1
                if (((max_val - min_val) / min_val) * 100) < 50:
                    within_50_perc_plus1hr_buffer += 1

        print(f"{total=}")
        print(f"Within 100 percentage accuracy: {(within_100_perc / total) * 100}%")
        print(f"Within 200 percentage accuracy: {(within_200_perc / total) * 100}%")
        print(f"Within 100 percentage accuracy + buffer: {(within_100_perc_plus1hr_buffer / total) * 100}%")
        print(f"Within 50 percentage accuracy + buffer: {(within_50_perc_plus1hr_buffer / total) * 100}%")
        print(f"Metric 1: {(my_metric_1 / total) * 100}%")
        run["test/within_100_perc"].append((within_100_perc / total) * 100)
        run["test/within_200_perc"].append((within_200_perc / total) * 100)
        run["test/buffer_within_100_perc"].append((within_100_perc_plus1hr_buffer / total) * 100)
        run["test/buffer_within_50_perc"].append((within_50_perc_plus1hr_buffer / total) * 100)
        run["test/my_metric_1"].append((my_metric_1 / total) * 100)
        scores.append((within_100_perc_plus1hr_buffer / total) * 100)
            
    print(loss_by_fold)
    print(np.mean(loss_by_fold))
    
    
    # Save best model
    torch.save(best_model.state_dict(), 'best_model.pt')
    
    # Final validation :)
    final_loss = []
    X_test = []
    y_pred = []
    y_actual = []
    with torch.no_grad():
        for X, y in last_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            final_loss.append(loss.item())
            X_test.extend(X)
            y_pred.extend(pred.flatten())
            y_actual.extend(y)
    
    r_value = pearsonr(y_pred, y_actual)
    print(f"Model number {model_idx} / {total_models} had a pearson r of")
    print(r_value)
    print("and a loss of ", np.mean(final_loss))
    
    run.stop()

if __name__ == '__main__':
    app.run(main)
