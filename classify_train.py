import pandas as pd
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
import neptune.integrations.optuna as npt_utils
import transformations
import smogn
from torchmetrics.classification import MulticlassF1Score
import imblearn.over_sampling
import optuna
from optuna.trial import TrialState

import config_file
import read_db
from model import nn_model, classify_model


flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_boolean('shuffle', True,'Shuffle training/validation set')
flags.DEFINE_float('oversample', 0.4, 'Oversampling factor')
flags.DEFINE_float('undersample', 0.8, 'Undersampling factor')
flags.DEFINE_integer('n_jobs', 10_000, 'Number of jobs to run on')

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

def get_class_labels(y, threshold):
    np_array = np.where(y > threshold, 1, 0)
    return np_array

def to_one_hot(np_array, num_classes=2):
    one_hot = nn.functional.one_hot(torch.from_numpy(np_array), num_classes=num_classes)
    return one_hot


def create_dataloaders(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        shuffle=FLAGS.shuffle)

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

def count_classes(y):
    count = [0, 0]
    for i in range(y.shape[0]):
        count[y[i]] += 1
    return count


def balance_dataset(X, y):
    # print(count_classes(y))
    # print(pd.DataFrame(X).describe())

    # X, y = imblearn.over_sampling.SMOTE().fit_resample(X, y)
    over = imblearn.over_sampling.SMOTE(sampling_strategy=FLAGS.oversample)
    under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=FLAGS.undersample)
    steps = [('o', over), ('u', under)]
    pipeline = imblearn.pipeline.Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)

    # print(count_classes(y))
    # print(pd.DataFrame(X).describe())
    return X, y


def model_performance(model, X, y):
    pred = model(X)
    correct_pred = [0, 0]
    total_pred = [0, 0]
    for i in range(pred.shape[0]):
        pred_class = torch.argmax(pred[i]).item()
        true_class = torch.argmax(y[i]).item()
        if pred_class == true_class:
            correct_pred[true_class] += 1
        total_pred[true_class] += 1
    return sum(correct_pred) / sum(total_pred)

def define_model(trial, num_features):
    in_features = num_features
    n_layers = trial.suggest_int("n_layers", 1, 3)
    activ_fn = trial.suggest_categorical("activ_fn", ["relu", "tanh", "leaky_relu"])
    if activ_fn == "relu":
        activ_fn = nn.ReLU()
    elif activ_fn == "tanh":
        activ_fn = nn.Tanh()
    elif activ_fn == "sigmoid":
        activ_fn = nn.Sigmoid()
    elif activ_fn == "leaky_relu":
        activ_fn = nn.LeakyReLU()
    layers = []

    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activ_fn)
        p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 2))
    return nn.Sequential(*layers)

def feature_options(features):
    if features == "queue":
        return ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue"]
    elif features == "request":
        return ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes"]
    elif features == "running":
        return ["jobs_running", "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    elif features == "memory":
        return ["req_mem", "memory_ahead_queue", "memory_running"]
    elif features == "cpu":
        return ["req_cpus", "cpus_ahead_queue", "cpus_running"]
    elif features == "all":
        return ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
                "time_limit_ahead_queue", "priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
                "jobs_running", "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    elif features == "job_count":
        return ["priority", "jobs_ahead_queue", "jobs_running"]
def objective(trial):
    X, y_one_hot, feature_mapping_dict = load_data()
    features = trial.suggest_categorical("features", ['memory', 'cpu', 'job_count', 'queue', 'running', 'request', 'all'])
    chosen_features = feature_options(features)
    num_features = len(chosen_features)
    feature_idxs = []
    for feature in chosen_features:
        feature_idxs.append(feature_mapping_dict[feature])
    X = X[:, feature_idxs]
    train_dataloader, test_dataloader = create_dataloaders(X, y_one_hot)
    model = define_model(trial, num_features)

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam'])
    # epochs = trial.suggest_int('epochs', 10, 100)
    epochs = 25
    if optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Run training loop
    train_loss_by_epoch = []
    test_loss_by_epoch = []
    classes = ["Under 10min", "Over10min"]
    for epoch in range(epochs):
        correct_pred = [0, 0]
        total_pred = [0, 0]
        train_loss = []
        test_loss = []
        model.train()
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

        model.eval()
        for X, y in test_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss.append(loss.item())
                for i in range(pred.shape[0]):
                    pred_class = torch.argmax(pred[i]).item()
                    true_class = torch.argmax(y[i]).item()
                    if pred_class == true_class:
                        correct_pred[true_class] += 1
                    total_pred[true_class] += 1

        # print(f"Epoch = {epoch}, Train_loss = {np.mean(train_loss):.2f}, Test Loss = {np.mean(test_loss):.5f}")
        train_loss_by_epoch.append(np.mean(train_loss))
        test_loss_by_epoch.append(np.mean(test_loss))
        # for i in range(len(correct_pred)):
        #     print(f"{classes[i]} accuracy: {correct_pred[i] / total_pred[i]}")
        total_acc = sum(correct_pred) / sum(total_pred)
        # print(f"Total accuracy: {total_acc:.4f}")
        trial.report(total_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return total_acc


def load_data():
    num_jobs = FLAGS.n_jobs
    read_all = True if num_jobs == 0 else False

    df = read_db.read_to_df(table="new_jobs_all", read_all=read_all, jobs=num_jobs)
    y = df["planned"].to_numpy()
    y = get_class_labels(y, threshold=600)
    X = df.drop(["planned"], axis=1)
    X = X._get_numeric_data()
    feature_mapping_dict = {}
    for feature_name in X.columns:
        feature_mapping_dict[feature_name] = X.columns.get_loc(feature_name)
    X = X.to_numpy().astype(np.float32)

    X, y = balance_dataset(X, y)
    y_one_hot = to_one_hot(y, num_classes=2).numpy()
    return X, y_one_hot, feature_mapping_dict

def start_trials():
    run_study = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
        tags=["classify"]
    )
    neptune_callback = npt_utils.NeptuneCallback(run_study)


    study = optuna.create_study(direction='maximize', study_name='namez')
    study.optimize(objective, n_jobs=16, n_trials=50, timeout=500, callbacks=[neptune_callback])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    run_study.stop()

def main(argv):
    start_trials()


if __name__ == '__main__':
    app.run(main)
