import numpy as np
import pandas as pd
import sqlalchemy
import time
import config_file
from tqdm import tqdm

def create_engine():
    db_config = {
        "dbname": "sacctdata",
        "user": "postgres",
        "password": config_file.postgres_password,
        "host": "slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
        "port": "5432"
    }

    engine = sqlalchemy.create_engine(
        f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
    return engine

if __name__ == '__main__':
    print("Connecting to database")
    engine = create_engine()
    df = pd.read_sql_query("SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw FROM jobs_2021_2025_05_02 ORDER BY eligible", engine)
    
    print("Database loaded")
    df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['end_time'] = df['end_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    np_array = df.to_numpy()

    JOB_ID = 0
    ELIGIBLE = 1
    START_TIME = 2
    END_TIME = 3
    REQ_CPUS = 4
    REQ_MEM = 5
    REQ_NODES = 6
    TIME_LIMIT_RAW = 7

    # Initialize running_features array
    running_features = np.zeros((np_array.shape[0], 5), dtype=np.float64)

    print("Starting")
    start = time.time()
    
    # Create a list of job intervals
    job_intervals = [(np_array[job_idx, START_TIME], np_array[job_idx, END_TIME], job_idx) for job_idx in range(np_array.shape[0]) if np_array[job_idx, START_TIME] < np_array[job_idx, END_TIME)]

    for job in tqdm(range(np_array.shape[0])):
        eligible_time = np_array[job, ELIGIBLE]

        # Create boolean mask for overlapping jobs
        overlapping_indices = [interval[2] for interval in job_intervals if interval[0] <= eligible_time < interval[1] and interval[2] != job]
        
        if overlapping_indices:
            overlapping_indices = np.array(overlapping_indices)
            running_features[job, 0] = overlapping_indices.size
            running_features[job, 1] = np.sum(np_array[overlapping_indices, REQ_CPUS])
            running_features[job, 2] = np.sum(np_array[overlapping_indices, REQ_MEM])
            running_features[job, 3] = np.sum(np_array[overlapping_indices, REQ_NODES])
            running_features[job, 4] = np.sum(np_array[overlapping_indices, TIME_LIMIT_RAW])
    
    end = time.time()
    print("Finished main iterations, moving on to adding to df")
    print(end - start)
    
    new_feature_names = ['jobs_running', 'cpus_running', 'nodes_running', 'memory_running']
    for idx, column_name in enumerate(new_feature_names):
        df[column_name] = running_features[:, idx] 
        
    print("Data added to df")
    df.to_csv("data_with_running.csv")
