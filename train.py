from absl import app, flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import read_db
from model import nn_model


flags.DEFINE_boolean('cuda', False, 'Whether to use cuda.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_epochs', 100, 'Number of Epochs')
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
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # First step: converting to tensor
    x_train_to_tensor = torch.from_numpy(x_train).to(torch.float32)
    y_train_to_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_to_tensor = torch.from_numpy(x_test).to(torch.float32)
    y_test_to_tensor = torch.from_numpy(y_test).to(torch.float32)

    # Second step: Creating TensorDataset for Dataloader
    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    test_dataset = TensorDataset(x_test_to_tensor, y_test_to_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size)
    return train_dataloader, test_dataloader


def main(argv):
    feature_names = ["time_limit_raw", "priority", "req_cpus", "req_mem", "req_nodes"]
    num_features = len(feature_names)


    df = read_db.read_to_df()
    np_array = df.to_numpy()
    feature_indices = get_feature_indices(df, feature_names)
    target_index = get_planned_target_index(df)

    X, y = np_array[:, feature_indices], np_array[:, target_index]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    train_dataloader, test_dataloader = create_dataloaders(X, y)

    model = nn_model(num_features)

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(params=model.parameters(), lr=FLAGS.learning_rate)

    train_loss_by_epoch = []
    test_loss_by_epoch = []

    for epoch in tqdm(range(FLAGS.num_epochs)):
        train_loss = []
        test_loss = []
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())

        for X, y in test_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                test_loss.append(loss.item())

        print(f"Epoch = {epoch}, Train_loss = {np.mean(train_loss):.2f}, Test Loss = {np.mean(test_loss):.5f}")
        train_loss_by_epoch.append(np.mean(train_loss))
        test_loss_by_epoch.append(np.mean(test_loss))

    plt.plot(train_loss_by_epoch)
    plt.plot(test_loss_by_epoch)
    plt.legend(['Train_loss', 'Test loss'])
    plt.show()

if __name__ == '__main__':
    app.run(main)
