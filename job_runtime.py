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
from sklearn.tree import DecisionTreeRegressor
import pickle


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
    
    df["elapsed"] = df["elapsed"] / 60
    
    train_dataloader, test_dataloader = get_dataloaders(df, feature_names, target)
    
    
    reg = DecisionTreeRegressor()
    tree_predictions = []
    tree_y = []
    
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    X_train = train_df[feature_names].to_numpy()
    y_train = train_df[target].to_numpy()
    X_test = test_df[feature_names].to_numpy()
    y_test = test_df[target].to_numpy()
    
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    
    reg.fit(X_train, y_train)
    
    pred = reg.predict(X_test)
       
    print(pred.shape, y_test.shape)
    
    r = pearsonr(pred, y_test)
    mses = ((pred - y_test) ** 2).mean()
    print(r)
    print(mses)
    
    m, b = np.polyfit(pred, y_test, 1)
    plt.scatter(pred, y_test)
    plt.plot(pred, m * pred + b, "r-")
    plt.xlabel("y pred")
    plt.ylabel("y test")
    plt.show()
    
    df["predicted_run_time"] = reg.predict(df[feature_names])
    
    with open('decision_tree_regressor.pkl', 'wb') as f:
        pickle.dump(reg, f)
    