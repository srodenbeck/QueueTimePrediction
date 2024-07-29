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
import smogn
from scipy.stats import pearsonr
import transformations
import pandas as pd
import shap
from optuna.samplers import TPESampler
import optuna
from optuna.trial import TrialState
import neptune.integrations.optuna as npt_utils
import time

import classify_train
import config_file
import read_db
from model import nn_model



flags.DEFINE_integer('n_jobs', 0, 'Number of jobs to train on.')
flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of Epochs')
# flags.DEFINE_enum('loss', 'smooth_l1_loss', ['mse_loss', 'l1_loss', 'smooth_l1_loss'], 'Loss function')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
flags.DEFINE_integer('hl1', 32, 'Hidden layer 1 dim')
flags.DEFINE_integer('hl2', 32, 'Hidden layer 1 dim')
flags.DEFINE_float('dropout', 0.225, 'Dropout rate')
flags.DEFINE_boolean('transform', True,'Use transformations on features')
flags.DEFINE_enum('activ', 'leaky_relu', ['relu', 'leaky_relu', 'elu', 'gelu'], 'Activation function')
flags.DEFINE_boolean('shuffle', False,'Shuffle training/validation set')
flags.DEFINE_boolean('only_10min_plus', True, 'Only include jobs with planned longer than 10 mintues')
flags.DEFINE_boolean('use_early_stopping', False, 'Whether or not to use early stopping')
flags.DEFINE_integer('early_stopping_patience', 10, 'Patience for early stopping')
flags.DEFINE_boolean('condense_same_times', False, 'Whether or not to remove jobs submitted back to back, bar the first job')

FLAGS = flags.FLAGS

gl_custom_loss = {
    "train_within_1hr_correct":  0,
    "train_within_1hr_total":  0,
    "test_within_1hr_correct":  0,
    "test_within_1hr_total":  0,
    "val_within_1hr_correct": 0,
    "val_within_1hr_total": 0
}

gl_X_rows = None
gl_y_rows = None
gl_np_array = None
gl_num_features = None
gl_feature_names = None
gl_tscv = None
gl_run = None
gl_best_score = -1


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
        X_train, X_test = transformations.scale_min_max(X_train, X_test)
        # X_train, X_test = transformations.scale_log(X_train, X_test)

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

    res = np.random.permutation(len(X_train))
    X_train = X_train[res]
    y_train = y_train[res]
    res = np.random.permutation(len(X_val))
    X_val = X_val[res]
    y_val = y_val[res]
    res = np.random.permutation(len(X_test))
    X_test = X_test[res]
    y_test = y_test[res]


    if FLAGS.transform:
        _, X_val = transformations.scale_min_max(X_train, X_val)
        X_train, X_test = transformations.scale_min_max(X_train, X_test)

        # _, X_Test = transformations.scale_min_max_test(X_test)
        # _, X_test = transformations.scale_log(X_train, X_test)
        # X_train, X_val = transformations.scale_log(X_train, X_val)

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

def define_model(num_features, style):
    input_dim = num_features
    if style == "small_wide":
        model = torch.nn.Sequential(nn.Linear(input_dim, 64),
                  nn.LeakyReLU(),
                  nn.Dropout(FLAGS.dropout),
                  nn.Linear(64, 32),
                  nn.LeakyReLU(),
                  nn.Dropout(FLAGS.dropout),
                  nn.Linear(32, 1))
    elif style == "small_thin":
        model = torch.nn.Sequential(nn.Linear(input_dim, 32),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 1))
    elif style == "large_wide":
        model = torch.nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(64, 32),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 1))
    elif style == "large_thin":
        model = torch.nn.Sequential(nn.Linear(input_dim, 32),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 8),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(8, 1))
    elif style == "very_large":
        model = torch.nn.Sequential(nn.Linear(input_dim, 96),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(96, 64),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(64, 32),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(32, 1))
    elif style == "idklol":
        model = torch.nn.Sequential(nn.Linear(input_dim, 32),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(32, 32),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 16),
                                    nn.LeakyReLU(),
                                    nn.Dropout(FLAGS.dropout),
                                    nn.Linear(16, 1))
    return model

def train_model(trial, is_ret_model=False):
    global gl_custom_loss
    global gl_best_score
    loss_by_fold = []
    scores = []

    run = gl_run
    tscv = gl_tscv
    np_array = gl_np_array
    X_rows = gl_X_rows
    y_rows = gl_y_rows
    num_features = gl_num_features
    feature_names = gl_feature_names

    loss_fn = trial.suggest_categorical("loss_fn", ["L1Loss", "SmoothL1Loss"])

    # train_dataloader, test_dataloader, val_dataloader = create_dataloaders(X_rows, y_rows)

    for train_index, test_index in tscv.split(np_array):
        train_dataloader, test_dataloader = create_dataloaders_tscv(X_rows, y_rows, train_index, test_index)

        # Create model
        # style = trial.suggest_categorical("architecture_style", ["small_wide", "small_thin", "large_wide", "large_thin", "very_large", "idklol"])
        style = trial.suggest_categorical("architecture_style", ["very_large"])

        model = define_model(num_features, style)
        # model = nn_model(num_features, hl_1, hl_2, FLAGS.dropout, FLAGS.activ)
        if loss_fn == "L1Loss":
            loss_fn = nn.L1Loss()
        elif loss_fn == "SmoothL1Loss":
            loss_fn = nn.SmoothL1Loss()
        elif loss_fn == "MSE":
            loss_fn = nn.MSELoss()
        optimizer = optim.Adam(params=model.parameters(), lr=FLAGS.lr)

        # Run training loop
        train_loss_by_epoch = []
        val_loss_by_epoch = []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(FLAGS.epochs):
            train_loss = []
            val_loss = []
            gl_custom_loss = dict.fromkeys(gl_custom_loss, 0)

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

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            print(f"Epoch = {epoch}, Train_loss = {avg_train_loss:.2f}, Validation Loss = {avg_val_loss:.5f}")
            train_loss_by_epoch.append(avg_train_loss)
            val_loss_by_epoch.append(avg_val_loss)
            run["train/loss"].append(avg_train_loss)
            run["valid/loss"].append(avg_val_loss)

        # Graphing and getting R2 value of model pred vs actual
        print("Graphing pred vs actual and calculating pearsons r")
        model.eval()
        y_pred = []
        y_actual = []
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

        # print(f"Average test loss of {avg_test_loss}")
        # bin_width = 0.1
        # bins = np.arange(min(absolute_percentage_error), max(absolute_percentage_error) + bin_width, bin_width)
        # plt.hist(absolute_percentage_error, bins=bins, density=True)
        # plt.xlabel('Absolute Percentage Error')
        # plt.ylabel('Probability Density')
        # plt.title("Density Histogram of Absolute Percentage Error of Test Data")
        # plt.ylim(0.0, 1.0)
        # plt.xlim(0.0, 1.0)
        # plt.show()

        # Average absolute percentage error
        # avg_ape = np.mean(absolute_percentage_error)
        # print("Mean absolute percentage error:", avg_ape * 100, "%")
        #
        # r_value = pearsonr(y_pred, y_actual)
        # print("pearson's r: ", r_value)
        # plt.scatter(y_pred, y_actual)
        # plt.xlabel("y predicted")
        # plt.ylabel("y")
        # plt.show()

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
        # explainer = shap.KernelExplainer(predict_function, background_data)
        # Calculate SHAP values
        # shap_values = explainer.shap_values(X_to_explain)
        # Visualize SHAP values for the first instance in the test set
        # shap.initjs()
        # shap.summary_plot(shap_values, X_to_explain, feature_names=feature_names)
        # shap.force_plot(explainer.expected_value, shap_values[0], test_data[0])

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

            if y_actual[z] < 1 * 60:
                if max_val - min_val < 20:
                    my_metric_1 += 1
            elif y_actual[z] < 5 * 60:
                if max_val - min_val < 60:
                    my_metric_1 += 1
            elif y_actual[z] < 12 * 60:
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
        score = (my_metric_1 / total) * 100

        if score > gl_best_score:
            torch.save(model.state_dict(), "aus_reg_model.pt")
            gl_run["best/model"].append(style)
            gl_run["best/loss"].append(loss_fn)
            gl_best_score = score

    return score



def calculate_custom_loss(pred, y, train_test_val):
    global gl_custom_loss
    # Calculating loss of how many target values were within 60 minutes of predicted values
    sub_tensor = torch.sub(pred.flatten(), y)
    binary_within = torch.where(torch.abs(sub_tensor) < 60, 1, 0)
    pred[pred < 0] = 0
    gl_custom_loss[f"{train_test_val}_within_1hr_total"] += binary_within.shape[0]
    gl_custom_loss[f"{train_test_val}_within_1hr_correct"] += binary_within.sum().item()


def scale_greedy_users(df):
    scale = 0.2
    top_accs = config_file.top_50_accs
    print(f"{top_accs=}")

    pd.options.mode.chained_assignment = None
    match_df = df[df['account'].isin(top_accs)]
    match_df['time_limit_raw'] = match_df['time_limit_raw'] * scale
    match_df['req_cpus'] = match_df['req_cpus'] * scale
    match_df['req_mem'] = match_df['req_mem'] * scale
    match_df['req_nodes'] = match_df['req_nodes'] * scale

    df.update(match_df)
    return df

def load_data(read_all=True, num_jobs=0, feature_names=None):
    print("Reading from database")
    df = read_db.read_to_df(table="new_features_1milly_user", read_all=read_all, jobs=num_jobs, order_by="eligible",
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
    # df = df[df['planned'] > 10 * 60]
    print(f"Using {len(df)} jobs")

    df = scale_greedy_users(df)

    np_array = df.to_numpy()

    # Read in desired features and target columns to numpy arrays
    feature_indices = get_feature_indices(df, feature_names)
    target_index = get_planned_target_index(df)
    X_rows, y_rows = np_array[:, feature_indices], np_array[:, target_index]
    X_rows = X_rows.astype(np.float32)
    y_rows = y_rows.astype(np.float32)
    print("X shape:", X_rows.shape)
    print("y shape:", y_rows.shape)
    return X_rows, y_rows, np_array, num_features, feature_names

def objective(trial):
    score = train_model(trial, is_ret_model=False)
    return score

def start_trials():
    global gl_X_rows
    global gl_y_rows
    global gl_np_array
    global gl_num_features
    global gl_feature_names
    global gl_tscv
    global gl_run
    
    # Connect to neptune
    gl_run = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
        tags=["austin_regression"]
    )

    # Feature names to use in training
    feature_names = classify_train.feature_options("austin_hypo_4")
    num_jobs = FLAGS.n_jobs
    read_all = True if num_jobs == 0 else False

    gl_X_rows, gl_y_rows, gl_np_array, gl_num_features, gl_feature_names = load_data(read_all, num_jobs, feature_names)

    # Specified parameters to upload to neptune
    params = {
        'feature_names': str(feature_names),
        'num_features': gl_num_features,
        'num_jobs': num_jobs,
        'jobs_used': len(gl_np_array),
        'lr': FLAGS.lr,
        'batch_size': FLAGS.batch_size,
        'epochs': FLAGS.epochs,
        'optimizer': FLAGS.optimizer,
        'dropout': FLAGS.dropout,
        '10_min_plus': FLAGS.only_10min_plus
    }
    gl_run["parameters"] = params

    # Transformations
    gl_y_rows = gl_y_rows / 60
    
    n_splits = 5
    gl_tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(gl_np_array) // (2 * n_splits + 1))

    sampler = TPESampler(n_startup_trials=10)
    study = optuna.create_study(direction='maximize', study_name='namez', sampler=sampler)
    study.optimize(objective, n_trials=40)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    trial = study.best_trial

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    npt_utils.log_study_metadata(study, gl_run)

    gl_run.stop()

def main(argv):
    start_trials()

if __name__ == '__main__':
    app.run(main)
