# -*- coding: utf-8 -*-

import model
import read_db
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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


class JobDataset(Dataset):
    def __init__(self, data, feature_names, target):
        self.features = data[feature_names].values
        self.target = data[target].values
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
def get_dataloaders(df, feature_names, target, test_size=0.2, batch_size=32, shuffle=True):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=0)
    
    train_dataset = JobDataset(train_df, feature_names, target)
    test_dataset = JobDataset(test_df, feature_names, target)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader
    

if __name__ == "__main__":

    feature_names = ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
                     "par_total_nodes", "par_total_cpu", "par_cpu_per_node", "par_mem_per_node", "par_total_gpu"]
    target = "elapsed"
    
    
    df = read_db.read_to_df(table="jobs_everything_all_2")
    print("Finished reading from database")
    
    temp_df = df['partition'].map(partition_feature_dict).apply(pd.Series)
    temp_df = temp_df.fillna(1)
    
    # Rename the columns in the temporary dataframe
    temp_df.columns = ['par_total_nodes', 'par_total_cpu', 'par_cpu_per_node', 'par_mem_per_node', 'par_total_gpu']
    
    # Concatenate the original dataframe with the temporary dataframe
    df = pd.concat([df, temp_df], axis=1)
    df = df.fillna(1)
    
    # Log transform data
    for feat in feature_names:
        df[feat] = np.log1p(df[feat])
    
    train_dataloader, test_dataloader = get_dataloaders(df, feature_names, target)
    
    
    input_dim = len(feature_names)
    hl1 = 24
    hl2 = 12
    dropout = 0.15
    activ = "relu"
    model = model.job_model(input_dim, hl1, hl2, dropout, activ)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    epochs = 50

    train_loss_by_epoch = []
    test_loss_by_epoch = []
    
    for epoch in range(epochs):
        train_loss = []
        test_loss = []
    
        # Training
        model.train()
        for X, y in train_dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred.flatten(), y)
            loss.backward()
            # nn_utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            
            train_loss.append(loss.item())
    
        # Evaluation/Validation
        model.eval()
        for X, y in test_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred.flatten(), y)
                test_loss.append(loss.item())
    
        avg_train_loss = np.mean(train_loss)
        avg_test_loss = np.mean(test_loss)
        
        train_loss_by_epoch.append(avg_train_loss)
        test_loss_by_epoch.append(avg_test_loss)
        
        print(f"epoch {epoch} / {epochs} - train loss of {avg_train_loss} - test loss of {avg_test_loss}")
        
    
    
    plt.plot(range(epochs), train_loss_by_epoch, "-b", label="Train Loss")
    plt.plot(range(epochs), test_loss_by_epoch, "-g", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss by Epoch")
    plt.legend(loc="upper right")
    plt.show()
    
    model.eval()
    y_pred = []
    y_actual = []
    for X, y in test_dataloader:
        pred = model(X)
        y_pred.extend(pred.flatten())
        y_actual.append(y)
    
    r_result = pearsonr(y_pred, y_actual)
    
    print(r_result)
    
    m, b = np.polyfit(y_pred, y_actual, 1)
    plt.scatter(y_pred, y_actual)
    plt.plot(y_pred, m * y_pred + b, "r-")
    plt.xlabel("y pred")
    plt.ylabel("y actual")
    plt.show()
    
        
        