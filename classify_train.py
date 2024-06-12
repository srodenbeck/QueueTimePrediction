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
import transformations
import smogn
from torchmetrics.classification import MulticlassF1Score
import imblearn.over_sampling
import optuna

import config_file
import read_db
from model import nn_model, classify_model


flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of Epochs')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam', 'adamw'], 'Optimizer algorithm')
flags.DEFINE_integer('hl1', 32, 'Hidden layer 1 dim')
flags.DEFINE_integer('hl2', 16, 'Hidden layer 1 dim')
flags.DEFINE_boolean('transform', False,'Use transformations on features')
flags.DEFINE_boolean('shuffle', True,'Shuffle training/validation set')
flags.DEFINE_float('oversample', 0.4, 'Oversampling factor')
flags.DEFINE_float('undersample', 0.8, 'Undersampling factor')

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

def get_class_labels(df, threshold):
    np_array = np.where(df['planned'] > threshold, 1, 0)
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
    print(count_classes(y))
    print(pd.DataFrame(X).describe())

    # X, y = imblearn.over_sampling.SMOTE().fit_resample(X, y)
    over = imblearn.over_sampling.SMOTE(sampling_strategy=FLAGS.oversample)
    under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=FLAGS.undersample)
    steps = [('o', over), ('u', under)]
    pipeline = imblearn.pipeline.Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)

    print(count_classes(y))
    print(pd.DataFrame(X).describe())
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


# def create_model()



def main(argv):
    feature_names = ["priority", "req_cpus", "req_mem", "jobs_ahead_queue", "memory_ahead_queue", "jobs_running", "memory_running"]
    num_features = len(feature_names)
    num_jobs = 100000
    read_all = True if num_jobs == 0 else False

    df = read_db.read_to_df(table="new_jobs_all", read_all=read_all, jobs=num_jobs)
    np_array = df.to_numpy()

    # Read in desired features and target columns to numpy arrays
    feature_indices = get_feature_indices(df, feature_names)
    X = np_array[:, feature_indices]
    X = X.astype(np.float32)

    y = get_class_labels(df, threshold=600)
    X, y = balance_dataset(X, y)
    y_one_hot = to_one_hot(y, num_classes=2).numpy()

    train_dataloader, test_dataloader = create_dataloaders(X, y_one_hot)

    model = classify_model(num_features, FLAGS.hl1, FLAGS.hl2)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

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
    classes = ["Under 10min", "Over10min"]
    correct_pred = [0, 0]
    total_pred = [0, 0]

    for epoch in range(FLAGS.epochs):
        train_loss = []
        test_loss = []
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

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

        print(f"Epoch = {epoch}, Train_loss = {np.mean(train_loss):.2f}, Test Loss = {np.mean(test_loss):.5f}")
        train_loss_by_epoch.append(np.mean(train_loss))
        test_loss_by_epoch.append(np.mean(test_loss))
        for i in range(len(correct_pred)):
            print(f"{classes[i]} accuracy: {correct_pred[i] / total_pred[i]}")
        print(f"Total accuracy: {sum(correct_pred) / sum(total_pred)}")


if __name__ == '__main__':
    app.run(main)
