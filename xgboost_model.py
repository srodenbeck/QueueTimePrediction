# -*- coding: utf-8 -*-

import xgboost as xgb
import read_db
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import pearsonr


if __name__ == "__main__":
    
    df = read_db.read_to_df("jobs_all_2")
    df = df[df['planned'] > 10 * 60]
    
    X = df[["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
            "time_limit_ahead_queue", "priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
            "eligible"]]
    y = df['planned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    
    reg = xgb.XGBRegressor(objective="reg:absoluteerror")
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    mae = np.mean(np.abs(y_pred - y_test))
    results = pearsonr(y_pred, y_test)
    print("Mean absolute error:", mae)
    print("Pearson r results:", results)
     