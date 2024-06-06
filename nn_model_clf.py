# -*- coding: utf-8 -*-


import torch
from torch import nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

INPUT_SIZE = 8

intervals = {
    "less than 5 minutes": 5 * 60,
    "less than 15 minutes": 15 * 60,
    "less than an hour": 60 * 60,
    "less than 4 hours": 4 * 60 * 60,
    "less than a day": 24 * 60 * 60,
    "more than a day": float('inf')  # Representing values more than a day
}

NUM_CLASSES = len(intervals)

def categorize_time(queue_time):
    for label, interval in intervals.items():
        if queue_time < interval:
            return label
    return "more than a day"

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"    
    )
    return device

device = get_device()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Avg loss: {test_loss:>8f} \n")


def get_tensors(path):
    df = pd.read_csv(path)
    df = df.iloc[8192:, :]
    print(df.shape)
    print(df.columns)
    
    df.reset_index(inplace=True)
    training = df.iloc[:90000, :]
    testing = df.iloc[90000:, :]
    X_train = training[['req_cpus', 'req_mem', 'req_nodes', 'time_limit_raw',
                       'jobs_running', 'cpus_running', 'nodes_running',
                       'memory_running']].values
    y_train = training['planned'].values
    X_test = testing[['req_cpus', 'req_mem', 'req_nodes', 'time_limit_raw',
                       'jobs_running', 'cpus_running', 'nodes_running',
                       'memory_running']].values
    y_test = testing['planned'].values
    
    y_train_categorical = [categorize_time(queue_time) for queue_time in y_train]
    y_test_categorical = [categorize_time(queue_time) for queue_time in y_test]
    


    scaler = MinMaxScaler()
    scaler.fit(X_train)    
    
    # Normalize training data
    X_train = scaler.transform(X_train)
    
    # Normalize testing data using scaler fitted on training data
    X_test = scaler.transform(X_test)
    
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_categorical)
    y_test_encoded = label_encoder.fit_transform(y_test_categorical)
    
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
#            nn.Linear(64, 64),
#            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__":
    
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = get_tensors("/Users/philipwisniewski/spyder/reu-p4/QueueTimePrediction/data_with_running.csv")
    
    # Training Dataset and Dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Testing Dataset and DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    INPUT_SIZE = len(['req_cpus', 'req_mem', 'req_nodes', 'time_limit_raw',
                       'jobs_running', 'cpus_running', 'nodes_running',
                       'memory_running'])
    
    model = NeuralNetwork().to(get_device())
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    
    print("\nStarting Training\n")
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Fiished Training!")
    
    
    
    