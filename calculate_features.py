import numpy as np
import pandas as pd
import sqlalchemy
import time
import config_file
from tqdm import tqdm
from intervaltree import IntervalTree, Interval

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

def read_to_np(engine):
        df = pd.read_sql_query("SELECT * FROM jobs ORDER BY RANDOM() LIMIT 1000", engine)
        np_array = df.to_numpy()
        return np_array


if __name__ == '__main__':
    engine = create_engine()
    df = pd.read_sql_query("SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw FROM jobs ORDER BY eligible", engine)
    tree = IntervalTree()
    # df = df[df.start_time != df.end_time]
    df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['end_time'] = df['end_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    np_array = df.to_numpy()
    np_array = np.append(np_array, np.int32(np.zeros([len(np_array), 1])), 1)

    jobs_running = np.zeros(8192)

    JOB_ID = 0
    ELIGIBLE = 1
    START_TIME = 2
    END_TIME = 3
    REQ_CPUS = 4
    REQ_MEM = 5
    REQ_NODES = 6
    TIME_LIMIT_RAW = 7
    
    # Columns: jobs_running, cpus_running, nodes_running, memory_running
    running_features = np.zeros((np_array.shape[0], 5))
    
    max_jobs = -1
    print("Starting")
    start = time.time()
    for job in tqdm(range(np_array.shape[0])):
        if np_array[job, START_TIME] < np_array[job, END_TIME]:
            tree[np_array[job, START_TIME]:np_array[job, END_TIME]] = np_array[job, 0]

    for job in tqdm(range(np_array.shape[0])):
        jobs_running[:] = 0
        for idx, overlapping in enumerate(tree[np_array[job, ELIGIBLE]]):
            if overlapping[2] != np_array[job, JOB_ID]:
                jobs_running[idx] = overlapping[2]
                # Increment number of jobs running
                running_features[job, 0] += 1
                # Add number of cpus
                running_features[job, 1] += np_array[job, REQ_CPUS]
                # Add memory
                running_features[job, 2] += np_array[job, REQ_MEM]
                # Add nodes
                running_features[job, 3] += np_array[job, REQ_NODES]
                # Add running time remaining
                running_features[job, 4] += np_array[job, TIME_LIMIT_RAW]
                
    
    end = time.time()
    print("Finished main iterations, moving on to adding to df")
    print(end - start)
    
    new_feature_names = ['jobs_running', 'cpus_running', 'nodes_running', 'memory_running']
    for idx, column_name in enumerate(new_feature_names):
        df[column_name] = running_features[:, idx] 
        
    print("Data added to df")
    df.to_csv("data_with_running.csv")
    


