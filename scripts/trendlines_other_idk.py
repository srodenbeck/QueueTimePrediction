import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

import read_db

np.random.seed(0)

def time_to_seconds(time_str):
    """
    Function to convert time_str which is in format D-HH-MM-SS to seconds
    """
    if '-' in time_str:
        d_hms = time_str.split('-')
        days = int(d_hms[0])
        h, m, s = map(int, d_hms[1].split(':'))
    else:
        days = 0
        h, m, s = map(int, time_str.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s

def memory_to_gigabytes(memory_str):
    """
    Function to convert memory_str which is in format ____U where U is unit
    into an integer representing size in gigabytes
    """
    if memory_str.endswith('T'):
        return float(memory_str[:-1]) * 1024
    elif memory_str.endswith('G'):
        return float(memory_str[:-1])
    elif memory_str.endswith('M'):
        return float(memory_str[:-1]) / 1024
    else:
        raise ValueError(f"Unknown memory unit in {memory_str}")

# Dictionary to convert partition into enum
partition_dict = {
    "standard": 1,
    "shared": 2,
    "debug": 3,
    "gpu": 4,
    "highmem": 5,
    "gpu-debug": 6,
    "wide": 7,
    "benchmarking": 8,
    "wholenode": 9,
    "azure": 10
}


if __name__=="__main__":
    
    # Read file
    print("Reading in database")
    df = read_db.read_to_df(table="new_jobs_all", read_all=True)
    
    feature_names = ["cpus_ahead_queue", "jobs_ahead_queue"
                     "memory_ahead_queue", "nodes_ahead_queue", "time_limit_ahead_queue",
                     "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    num_features = len(feature_names)
    target = 'planned'
    
    # Path to where images will be stored
    path_root = "../graphs/TrendLines/"
    path_end = "_5_minute_plus.png"
    
    # y = df['PlannedRaw']
    
    print("Only keeping jobs greater than 5 minutes")
    
    # Only show jobs with wait time greater than a minute
    df = df[df['planned'] > 60 * 5]
    
    # Convert data to hours
    df['planned'] = df['planned'] / 3600
    df = df[df['planned'] < 200 ]
    y = df['planned']
    
    
    print("Making graphs")
    # Make graph for each independent variable
    for name in feature_names:
        print(name)
        plt.figure(figsize=(10,6))
        f, ax = plt.subplots()
        df = df.sort_values(by=name)
        # moved above
        # df_copy = df[df['PlannedRaw'] > 60]
    
        # Index can be adjusted to only graph first X% of data
        # index = int(0.99 * len(df))
        index = len(df)
        X = df[name]
        
        m, b = np.polyfit(X, y, 1)
        plt.plot(X, m * X + b, color="red")
    
        plt.scatter(X, y)
        plt.xlabel(name)
        plt.ylabel('Planned (h)')
        plt.title(f"{name} vs Planned (Wait longer than 5 minutes)")
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, y)
        plt.plot(X, slope*X + intercept, "r-")
        ax.text(0.1, 0.9, f"R2 value: {r_value * r_value}", transform=ax.transAxes)
        # plt.annotate("r-squared = {:.3f}".format(r_value), (0,1))
        # plt.annotate("p-value = {:.5f}".format(p_value), (0,0.9))
        # plt.savefig(path_root + name + path_end)
        plt.show()
        

