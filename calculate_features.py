import numpy as np
import pandas as pd
import sqlalchemy
from tqdm import tqdm
from intervaltree import IntervalTree

import config_file
import read_db

def calculate_running_features():
    """
    calculate_running_features()

    Calculates features relating to jobs currently running when another job is
    made eligible. Features include number of jobs running, the total number of
    cpus in use by running jobs, the total amount of memory being used by
    running jobs, the total amount of nodes being used by running jobs, and the
    combined timelimit for all running jobs.

    Parameters
    ----------
    engine : SQLALCHEMY ENGINE
        Instance of sqlalchemy engine to access postgres database.

    Returns
    -------
    None.

    """
    engine = read_db.create_engine()
    # Read in dataframe
    all_df = pd.read_sql_query("SELECT * FROM jobs_2024_05_02_1mil ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw, partition FROM jobs_2024_05_02_1mil ORDER BY eligible",
        engine)
    engine.dispose()
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
        "TIME_LIMIT_RAW": 7,
        "PARTITION": 8
    }

    # Columns: job_id, jobs_running, cpus_running, memory_running, nodes_running, time_limit_raw
    running_features = np.zeros((np_array.shape[0], 6))
    par_running_features = np.zeros((np_array.shape[0], 6))

    num_trees = 50
    tree_size = 100000
    tree_overlap = 10000


    # Creation of overlapping interval trees
    count = 0
    tree_idx = 0
    trees = []
    for i in range(num_trees):
        trees.append(IntervalTree())
    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        if np_array[job_idx, idx_dict["START_TIME"]] < np_array[job_idx, idx_dict["END_TIME"]]:
            trees[tree_idx][np_array[job_idx, idx_dict["START_TIME"]]:np_array[job_idx, idx_dict["END_TIME"]]] = job_idx
        # Make last tree size of 3 normal trees to prevent edge case issues
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tmp = sorted(trees[tree_idx])[-tree_overlap:]
            for interval in tmp:
                trees[tree_idx + 1][interval[0]:interval[1]] = interval[2]
            if tree_idx != 0:
                tmp = sorted(trees[tree_idx])[0:tree_overlap]
                for interval in tmp:
                    trees[tree_idx - 1][interval[0]:interval[1]] = interval[2]
            tree_idx += 1
            count = 0

    # Loop through jobs and add in data for all jobs whose trees overlap
    tree_idx = 0
    count = 0
    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        running_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        par_running_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        for overlapping in trees[tree_idx][np_array[job_idx, idx_dict["ELIGIBLE"]]]:
            if overlapping[2] != np_array[job_idx, idx_dict["JOB_ID"]]:
                jobs_running_idx = overlapping[2]
                running_features[job_idx, 1] += 1
                running_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]]
                running_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                running_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]]
                running_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]]
                if np_array[jobs_running_idx, idx_dict["PARTITION"]] == np_array[job_idx, idx_dict["PARTITION"]]:
                    par_running_features[job_idx, 1] += 1
                    par_running_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]]
                    par_running_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                    par_running_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]]
                    par_running_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]]
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tree_idx += 1
            count = 0

    # Update dataframe from results and upload to database
    new_df = pd.DataFrame(
        {"job_id": running_features[:, 0].astype(np.uint32), "jobs_running": running_features[:, 1].astype(np.uint32),
         "cpus_running": running_features[:, 2].astype(np.uint32),
         "memory_running": running_features[:, 3], "nodes_running": running_features[:, 4].astype(np.uint32),
         "time_limit_running": running_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    new_df = pd.DataFrame(
        {"job_id": par_running_features[:, 0].astype(np.uint32),
         "par_jobs_running": par_running_features[:, 1].astype(np.uint32),
         "par_cpus_running": par_running_features[:, 2].astype(np.uint32),
         "par_memory_running": par_running_features[:, 3],
         "par_nodes_running": par_running_features[:, 4].astype(np.uint32),
         "par_time_limit_running": par_running_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    engine = read_db.create_engine()
    all_df.to_sql('tmp4', engine, if_exists='replace', index=False)
    engine.dispose()


def calculate_queue_features():
    engine = read_db.create_engine()
    # Read in dataframe
    all_df = pd.read_sql_query("SELECT * FROM tmp4 ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw, partition FROM tmp4 ORDER BY eligible",
        engine)
    engine.dispose()

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
        "TIME_LIMIT_RAW": 7,
        "PARTITION": 8
    }

    # Columns: job_id, jobs_ahead_queue, cpus_ahead_queue, memory_ahead_queue, nodes_ahead_queue, time_limit_raw
    queue_features = np.zeros((np_array.shape[0], 6))
    par_queue_features = np.zeros((np_array.shape[0], 6))

    num_trees = 50
    tree_size = 100000
    tree_overlap = 10000

    # Creation of interval trees
    count = 0
    tree_idx = 0
    trees = []
    for i in range(num_trees):
        trees.append(IntervalTree())
    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        if np_array[job_idx, idx_dict["ELIGIBLE"]] != np_array[job_idx, idx_dict["START_TIME"]]:
            trees[tree_idx][np_array[job_idx, idx_dict["ELIGIBLE"]]:np_array[job_idx, idx_dict["START_TIME"]]] = job_idx
        # Make last tree size of 3 normal trees to prevent edge case issues
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tmp = sorted(trees[tree_idx])[-tree_overlap:]
            for interval in tmp:
                trees[tree_idx + 1][interval[0]:interval[1]] = interval[2]
            if tree_idx != 0:
                tmp = sorted(trees[tree_idx])[0:tree_overlap]
                for interval in tmp:
                    trees[tree_idx - 1][interval[0]:interval[1]] = interval[2]
            tree_idx += 1
            count = 0

    # Loop through jobs and add in data for all jobs whose trees overlap
    tree_idx = 0
    count = 0
    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        queue_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        par_queue_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        for overlapping in trees[tree_idx][np_array[job_idx, idx_dict["ELIGIBLE"]]]:
            if overlapping[2] != np_array[job_idx, idx_dict["JOB_ID"]]:
                jobs_running_idx = overlapping[2]
                queue_features[job_idx, 1] += 1
                queue_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]]
                queue_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                queue_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]]
                queue_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]]
                if np_array[jobs_running_idx, idx_dict["PARTITION"]] == np_array[job_idx, idx_dict["PARTITION"]]:
                    par_queue_features[job_idx, 1] += 1
                    par_queue_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]]
                    par_queue_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                    par_queue_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]]
                    par_queue_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]]
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tree_idx += 1
            count = 0

    # Update dataframe from results and upload to database
    new_df = pd.DataFrame(
        {"job_id": queue_features[:, 0].astype(np.uint32), "jobs_ahead_queue": queue_features[:, 1].astype(np.uint32),
         "cpus_ahead_queue": queue_features[:, 2].astype(np.uint32),
         "memory_ahead_queue": queue_features[:, 3], "nodes_ahead_queue": queue_features[:, 4].astype(np.uint32),
         "time_limit_ahead_queue": queue_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    new_df = pd.DataFrame(
        {"job_id": par_queue_features[:, 0].astype(np.uint32), "par_jobs_ahead_queue": par_queue_features[:, 1].astype(np.uint32),
         "par_cpus_ahead_queue": par_queue_features[:, 2].astype(np.uint32),
         "par_memory_ahead_queue": par_queue_features[:, 3], "par_nodes_ahead_queue": par_queue_features[:, 4].astype(np.uint32),
         "par_time_limit_ahead_queue": par_queue_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    engine = read_db.create_engine()
    all_df.to_sql('tmp5', engine, if_exists='replace', index=False)
    engine.dispose()

def calculate_higher_priority_queue_features():
    """
    calculate_higher_priority_queue_features()

    Calculates features relating to jobs currently in the queue when another job
    is made eligible with a higher priority than said job. Features include 
    number of jobs queued, the total number of
    cpus in use by queued jobs, the total amount of memory being used by
    queued jobs, the total amount of nodes being used by queued jobs, and the
    combined timelimit for queued jobs with a higher priority.

    Parameters
    ----------
    None.
    
    Returns
    -------
    None.

    """
    engine = read_db.create_engine()
    # Read in dataframe
    print("Reading in dataframes")
    all_df = pd.read_sql_query("SELECT * FROM tmp5 ORDER BY eligible ", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw, priority FROM tmp5 ORDER BY eligible",
        engine)
    engine.dispose()
    df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['end_time'] = df['end_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    np_array = df.to_numpy()

    print("Dataframes successfully read into numpy arrays")
    
    idx_dict = {
        "JOB_ID": 0,
        "ELIGIBLE": 1,
        "START_TIME": 2,
        "END_TIME": 3,
        "REQ_CPUS": 4,
        "REQ_MEM": 5,
        "REQ_NODES": 6,
        "TIME_LIMIT_RAW": 7,
        "PRIORITY": 8
    }

    # Columns: job_id, jobs_ahead_queue_priority, cpus_ahead_queue_priority, memory_ahead_queue_priority, nodes_ahead_queue_priority, time_limit_raw_queue_priority
    queue_features = np.zeros((np_array.shape[0], 6))
    engine.dispose()
    num_trees = 50
    tree_size = 100000
    tree_overlap = 10000

    # Creation of interval trees
    count = 0
    tree_idx = 0
    trees = []
    for i in range(num_trees):
        trees.append(IntervalTree())
    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        if np_array[job_idx, idx_dict["ELIGIBLE"]] != np_array[job_idx, idx_dict["START_TIME"]]:
            trees[tree_idx][np_array[job_idx, idx_dict["ELIGIBLE"]]:np_array[job_idx, idx_dict["START_TIME"]]] = job_idx
        # Make last tree size of 3 normal trees to prevent edge case issues
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tmp = sorted(trees[tree_idx])[-tree_overlap:]
            for interval in tmp:
                trees[tree_idx + 1][interval[0]:interval[1]] = interval[2]
            if tree_idx != 0:
                tmp = sorted(trees[tree_idx])[0:tree_overlap]
                for interval in tmp:
                    trees[tree_idx - 1][interval[0]:interval[1]] = interval[2]
            tree_idx += 1
            count = 0

    # Loop through jobs and add in data for all jobs whose trees overlap
    tree_idx = 0
    count = 0
    for job_idx in tqdm(range(np_array.shape[0])):
        count += 1
        queue_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        for overlapping in trees[tree_idx][np_array[job_idx, idx_dict["ELIGIBLE"]]]:
            if overlapping[2] != np_array[job_idx, idx_dict["JOB_ID"]]:
                jobs_running_idx = overlapping[2]
                # If job's priority is greater than current job's priority
                # Used to calculate features only for jobs with higher priority
                if np_array[jobs_running_idx, idx_dict["PRIORITY"]] > np_array[job_idx, idx_dict["PRIORITY"]]:
                    queue_features[job_idx, 1] += 1
                    queue_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]].astype(np.uint32)
                    queue_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                    queue_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]].astype(np.uint32)
                    queue_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]].astype(np.uint32)
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tree_idx += 1
            count = 0
    
    # Update dataframe from results and upload to database
    new_df = pd.DataFrame(
        {"job_id": queue_features[:, 0].astype(np.uint32), "jobs_ahead_queue_priority": queue_features[:, 1].astype(np.uint32),
         "cpus_ahead_queue_priority": queue_features[:, 2].astype(np.uint32),
         "memory_ahead_queue_priority": queue_features[:, 3], "nodes_ahead_queue_priority": queue_features[:, 4].astype(np.uint32),
         "time_limit_ahead_queue_priority": queue_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    engine = read_db.create_engine()
    all_df.to_sql('jobs_everything', engine, if_exists='replace', index=False)
    engine.dispose()


if __name__ == '__main__':
    # engine = read_db.create_engine()
    calculate_running_features()
    calculate_queue_features()
    calculate_higher_priority_queue_features()
