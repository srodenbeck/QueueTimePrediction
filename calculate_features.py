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

def calculate_running_features(engine):
    all_df = pd.read_sql_query("SELECT * FROM jobs_2021_2025_05_02 ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw FROM jobs_2021_2025_05_02 ORDER BY eligible",
        engine)
    df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['end_time'] = df['end_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    df['start_time'] = df['start_time'] - 1631807119
    df['end_time'] = df['end_time'] - 1631807119
    df['eligible'] = df['eligible'] - 1631807119
    np_array = df.to_numpy()

    # Dict to be used for indexing columns of np_array.
    idx_dict = {
        "JOB_ID": 0,
        "ELIGIBLE": 1,
        "START_TIME": 2,
        "END_TIME": 3,
        "REQ_CPUS": 4,
        "REQ_MEM": 5,
        "REQ_NODES": 6,
        "TIME_LIMIT_RAW": 7
    }

    # Columns: job_id, jobs_running, cpus_running, memory_running, nodes_running, time_limit_raw
    running_features = np.zeros((np_array.shape[0], 6))

    num_trees = 50
    tree_size = 100000
    tree_overlap = 10000

    count = 0
    tree_idx = 0
    trees = []
    for i in range(num_trees):
        trees.append(IntervalTree())

    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        if np_array[job_idx, idx_dict["START_TIME"]] < np_array[job_idx, idx_dict["END_TIME"]]:
            trees[tree_idx][np_array[job_idx, idx_dict["START_TIME"]].astype(np.int32):np_array[job_idx, idx_dict["END_TIME"]].astype(np.int32)] = job_idx
        if count == tree_size:
            tmp = sorted(trees[tree_idx])[-tree_overlap:]
            for interval in tmp:
                trees[tree_idx + 1][interval[0]:interval[1]] = interval[2]
            if tree_idx != 0:
                tmp = sorted(trees[tree_idx])[0:tree_overlap]
                for interval in tmp:
                    trees[tree_idx - 1][interval[0]:interval[1]] = interval[2]
            tree_idx += 1
            count = 0

    tree_idx = 0
    for job in tqdm(range(np_array.shape[0])):
        count += 1
        running_features[job, 0] = np_array[job, idx_dict["JOB_ID"]]
        for overlapping in trees[tree_idx][np_array[job, idx_dict["ELIGIBLE"]]]:
            if overlapping[2] != np_array[job, idx_dict["JOB_ID"]]:
                jobs_running_idx = overlapping[2]
                running_features[job, 1] += 1
                running_features[job, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]].astype(np.uint32)
                running_features[job, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                running_features[job, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]].astype(np.uint32)
                running_features[job, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]].astype(np.uint32)
        if count == tree_size:
            tree_idx += 1
            count = 0


    new_df = pd.DataFrame(
        {"job_id": running_features[:, 0].astype(np.uint32), "jobs_running": running_features[:, 1].astype(np.uint32),
         "cpus_running": running_features[:, 2].astype(np.uint32),
         "memory_running": running_features[:, 3], "nodes_running": running_features[:, 4].astype(np.uint32),
         "time_limit_running": running_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    all_df.to_sql('new_jobs_odd', engine, if_exists='replace', index=False)


def calculate_queue_features(engine):
    all_df = pd.read_sql_query("SELECT * FROM jobs_2021_2025_05_02 ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw FROM jobs_2021_2025_05_02 ORDER BY eligible",
        engine)
    df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['end_time'] = df['end_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    np_array = df.to_numpy()

    idx_dict = {
        "JOB_ID": 0,
        "ELIGIBLE": 1,
        "START_TIME": 2,
        "END_TIME": 3,
        "REQ_CPUS": 4,
        "REQ_MEM": 5,
        "REQ_NODES": 6,
        "TIME_LIMIT_RAW": 7
    }

    # Columns: job_id, jobs_ahead_queue, cpus_ahead_queue, memory_ahead_queue, nodes_ahead_queue, time_limit_raw
    queue_features = np.zeros((np_array.shape[0], 6))

    num_trees = 50
    tree_size = 100000
    tree_overlap = 10000

    count = 0
    tree_idx = 0
    trees = []
    for i in range(num_trees):
        trees.append(IntervalTree())

    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        if np_array[job_idx, idx_dict["ELIGIBLE"]] != np_array[job_idx, idx_dict["START_TIME"]]:
            trees[tree_idx][np_array[job_idx, idx_dict["ELIGIBLE"]]:np_array[job_idx, idx_dict["START_TIME"]]] = job_idx
        if count == tree_size:
            tmp = sorted(trees[tree_idx])[-tree_overlap:]
            for interval in tmp:
                trees[tree_idx + 1][interval[0]:interval[1]] = interval[2]
            if tree_idx != 0:
                tmp = sorted(trees[tree_idx])[0:tree_overlap]
                for interval in tmp:
                    trees[tree_idx - 1][interval[0]:interval[1]] = interval[2]
            tree_idx += 1
            count = 0

    for job in tqdm(range(np_array.shape[0])):
        count += 1
        queue_features[job, 0] = np_array[job, idx_dict["JOB_ID"]]
        for overlapping in trees[tree_idx][np_array[job, idx_dict["ELIGIBLE"]]]:
            if overlapping[1] < np_array[job, idx_dict["START_TIME"]] and overlapping[2] != np_array[job, idx_dict["JOB_ID"]]:
                jobs_running_idx = overlapping[2]
                queue_features[job, 1] += 1
                queue_features[job, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]].astype(np.uint32)
                queue_features[job, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                queue_features[job, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]].astype(np.uint32)
                queue_features[job, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]].astype(np.uint32)
        if count == tree_size:
            tree_idx += 1
            count = 0

    new_df = pd.DataFrame(
        {"job_id": queue_features[:, 0].astype(np.uint32), "jobs_ahead_queue": queue_features[:, 1].astype(np.uint32),
         "cpus_ahead_queue": queue_features[:, 2].astype(np.uint32),
         "memory_ahead_queue": queue_features[:, 3], "nodes_ahead_queue": queue_features[:, 4].astype(np.uint32),
         "time_limit_ahead_queue": queue_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    all_df.to_sql('new_jobs', engine, if_exists='replace', index=False)



if __name__ == '__main__':
    engine = create_engine()
    calculate_running_features(engine)
    # calculate_queue_features(engine)
