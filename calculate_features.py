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
    all_df = pd.read_sql_query("SELECT * FROM jobs_2021_2025_05_02 ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw, partition FROM jobs_2021_2025_05_02 ORDER BY eligible",
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
                # // jobs_running_idx = overlapping[2]
                # running_features[job_idx, 1] += 1
                # running_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]]
                # running_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                # running_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]]
                # running_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]]
                if np_array[overlapping[2], idx_dict["PARTITION"]] == np_array[job_idx, idx_dict["PARTITION"]]:
                    par_running_features[job_idx, 1] += 1
                    par_running_features[job_idx, 2] += np_array[overlapping[2], idx_dict["REQ_CPUS"]]
                    par_running_features[job_idx, 3] += np_array[overlapping[2], idx_dict["REQ_MEM"]]
                    par_running_features[job_idx, 4] += np_array[overlapping[2], idx_dict["REQ_NODES"]]
                    par_running_features[job_idx, 5] += np_array[overlapping[2], idx_dict["TIME_LIMIT_RAW"]]
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tree_idx += 1
            count = 0

    # Update dataframe from results and upload to database
    # new_df = pd.DataFrame(
    #     {"job_id": running_features[:, 0].astype(np.uint32), "jobs_running": running_features[:, 1].astype(np.uint32),
    #      "cpus_running": running_features[:, 2].astype(np.uint32),
    #      "memory_running": running_features[:, 3], "nodes_running": running_features[:, 4].astype(np.uint32),
    #      "time_limit_running": running_features[:, 5].astype(np.uint32)})
    # all_df.update(new_df)
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
                # jobs_running_idx = overlapping[2]
                # queue_features[job_idx, 1] += 1
                # queue_features[job_idx, 2] += np_array[jobs_running_idx, idx_dict["REQ_CPUS"]]
                # queue_features[job_idx, 3] += np_array[jobs_running_idx, idx_dict["REQ_MEM"]]
                # queue_features[job_idx, 4] += np_array[jobs_running_idx, idx_dict["REQ_NODES"]]
                # queue_features[job_idx, 5] += np_array[jobs_running_idx, idx_dict["TIME_LIMIT_RAW"]]
                if np_array[overlapping[2], idx_dict["PARTITION"]] == np_array[job_idx, idx_dict["PARTITION"]]:
                    par_queue_features[job_idx, 1] += 1
                    par_queue_features[job_idx, 2] += np_array[overlapping[2], idx_dict["REQ_CPUS"]]
                    par_queue_features[job_idx, 3] += np_array[overlapping[2], idx_dict["REQ_MEM"]]
                    par_queue_features[job_idx, 4] += np_array[overlapping[2], idx_dict["REQ_NODES"]]
                    par_queue_features[job_idx, 5] += np_array[overlapping[2], idx_dict["TIME_LIMIT_RAW"]]
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tree_idx += 1
            count = 0

    # Update dataframe from results and upload to database
    # new_df = pd.DataFrame(
    #     {"job_id": queue_features[:, 0].astype(np.uint32), "jobs_ahead_queue": queue_features[:, 1].astype(np.uint32),
    #      "cpus_ahead_queue": queue_features[:, 2].astype(np.uint32),
    #      "memory_ahead_queue": queue_features[:, 3], "nodes_ahead_queue": queue_features[:, 4].astype(np.uint32),
    #      "time_limit_ahead_queue": queue_features[:, 5].astype(np.uint32)})
    # all_df.update(new_df)
    new_df = pd.DataFrame(
        {"job_id": par_queue_features[:, 0].astype(np.uint32),
         "par_jobs_queue": par_queue_features[:, 1].astype(np.uint32),
         "par_cpus_queue": par_queue_features[:, 2].astype(np.uint32),
         "par_memory_queue": par_queue_features[:, 3], "par_nodes_queue": par_queue_features[:, 4].astype(np.uint32),
         "par_time_limit_queue": par_queue_features[:, 5].astype(np.uint32)})
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
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw, priority, partition  FROM tmp5 ORDER BY eligible",
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
        "PRIORITY": 8,
        "PARTITION": 9
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
                # // jobs_running_idx = overlapping[2]
                # If job's priority is greater than current job's priority
                # Used to calculate features only for jobs with higher priority
                if np_array[overlapping[2], idx_dict["PRIORITY"]] > np_array[job_idx, idx_dict["PRIORITY"]] and \
                        np_array[overlapping[2], idx_dict["PARTITION"]] == np_array[job_idx, idx_dict["PARTITION"]]:
                    queue_features[job_idx, 1] += 1
                    queue_features[job_idx, 2] += np_array[overlapping[2], idx_dict["REQ_CPUS"]]
                    queue_features[job_idx, 3] += np_array[overlapping[2], idx_dict["REQ_MEM"]]
                    queue_features[job_idx, 4] += np_array[overlapping[2], idx_dict["REQ_NODES"]]
                    queue_features[job_idx, 5] += np_array[overlapping[2], idx_dict["TIME_LIMIT_RAW"]]
        if count == tree_size and ((np_array.shape[0] - job_idx) > (3 * tree_size)):
            tree_idx += 1
            count = 0

    # Update dataframe from results and upload to database
    new_df = pd.DataFrame(
        {"job_id": queue_features[:, 0].astype(np.uint32),
         "par_jobs_ahead_queue": queue_features[:, 1].astype(np.uint32),
         "par_cpus_ahead_queue": queue_features[:, 2].astype(np.uint32),
         "par_memory_ahead_queue": queue_features[:, 3],
         "par_nodes_ahead_queue": queue_features[:, 4].astype(np.uint32),
         "par_time_limit_ahead_queue": queue_features[:, 5].astype(np.uint32)})
    all_df.update(new_df)
    engine = read_db.create_engine()
    all_df.to_sql('jobs_everything_all', engine, if_exists='replace', index=False)
    engine.dispose()


def calculate_user_features():
    engine = read_db.create_engine()
    # Read in dataframe
    all_df = pd.read_sql_query("SELECT * FROM new_features_1milly ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, user_id, eligible, req_cpus, req_mem, req_nodes, time_limit_raw FROM new_features_1milly ORDER BY eligible",
        engine)
    engine.dispose()

    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    np_array = df.to_numpy()
    idx_dict = {
        "JOB_ID": 0,
        "USER_ID": 1,
        "ELIGIBLE": 2,
        "REQ_CPUS": 3,
        "REQ_MEM": 4,
        "REQ_NODES": 5,
        "TIME_LIMIT_RAW": 6
    }

    usr_idx = {
        "ELIGIBLE": 0,
        "COUNT": 1,
        "REQ_CPUS": 2,
        "REQ_MEM": 3,
        "REQ_NODES": 4,
        "TIME_LIMIT_RAW": 5
    }

    user_features = np.zeros((np_array.shape[0], 6))
    user_dict = {}
    total_user_dict = {}

    for job_idx in tqdm(range(np_array.shape[0])):
        user_dict[np_array[job_idx, idx_dict["USER_ID"]]] = []
    for job_idx in tqdm(range(np_array.shape[0])):
        total_user_dict[np_array[job_idx, idx_dict["USER_ID"]]] = [None, 0, 0, 0.0, 0, 0]

    for job_idx in tqdm(range(np_array.shape[0])):
        user_id = np_array[job_idx, idx_dict["USER_ID"]]
        eligible = np_array[job_idx, idx_dict["ELIGIBLE"]]
        user_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        user_features[job_idx, 1] = total_user_dict[user_id][usr_idx["COUNT"]]
        user_features[job_idx, 2] = total_user_dict[user_id][usr_idx["REQ_CPUS"]]
        user_features[job_idx, 3] = total_user_dict[user_id][usr_idx["REQ_MEM"]]
        user_features[job_idx, 4] = total_user_dict[user_id][usr_idx["REQ_NODES"]]
        user_features[job_idx, 5] = total_user_dict[user_id][usr_idx["TIME_LIMIT_RAW"]]

        while user_dict[user_id] and eligible - user_dict[user_id][0][0] > (24 * 3600):
            total_user_dict[user_id][usr_idx["COUNT"]] -= 1
            total_user_dict[user_id][usr_idx["REQ_CPUS"]] -= user_dict[user_id][0][usr_idx["REQ_CPUS"]]
            total_user_dict[user_id][usr_idx["REQ_MEM"]] -= user_dict[user_id][0][usr_idx["REQ_MEM"]]
            total_user_dict[user_id][usr_idx["REQ_NODES"]] -= user_dict[user_id][0][usr_idx["REQ_NODES"]]
            total_user_dict[user_id][usr_idx["TIME_LIMIT_RAW"]] -= user_dict[user_id][0][usr_idx["TIME_LIMIT_RAW"]]
            user_dict[user_id].pop(0)

        user_dict[user_id].append([eligible,
                                   0,
                                   np_array[job_idx, idx_dict["REQ_CPUS"]],
                                   np_array[job_idx, idx_dict["REQ_MEM"]],
                                   np_array[job_idx, idx_dict["REQ_NODES"]],
                                   np_array[job_idx, idx_dict["TIME_LIMIT_RAW"]]])

        total_user_dict[user_id][usr_idx["COUNT"]] += 1
        total_user_dict[user_id][usr_idx["REQ_CPUS"]] += np_array[job_idx, idx_dict["REQ_CPUS"]]
        total_user_dict[user_id][usr_idx["REQ_MEM"]] += np_array[job_idx, idx_dict["REQ_MEM"]]
        total_user_dict[user_id][usr_idx["REQ_NODES"]] += np_array[job_idx, idx_dict["REQ_NODES"]]
        total_user_dict[user_id][usr_idx["TIME_LIMIT_RAW"]] += np_array[job_idx, idx_dict["TIME_LIMIT_RAW"]]

    new_df = pd.DataFrame(
        {"job_id": user_features[:, 0].astype(np.uint32),
         "user_jobs_past_day": user_features[:, 1].astype(np.uint32),
         "user_cpus_past_day": user_features[:, 2].astype(np.uint32),
         "user_memory_past_day": user_features[:, 3],
         "user_nodes_past_day": user_features[:, 4].astype(np.uint32),
         "user_time_limit_past_day": user_features[:, 5].astype(np.uint32)})

    result = pd.merge(all_df, new_df, how="left", on=["job_id"])
    engine = read_db.create_engine()
    result.to_sql('new_features_1milly_user', engine, if_exists='replace', index=False)
    engine.dispose()


def calculate_remaining_running_by_partition():
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

    # Columns: job_id, nodes, cpus, memory
    par_running_features = np.zeros((np_array.shape[0], 4))

    trees = []
    par_dict = {
        'standard': 0,
        'shared': 1,
        'wholenode': 2,
        'wide': 3,
        'gpu': 4,
        'highmem': 5,
        'azure': 6
    }

    max_per_par = {
        'standard': {'nodes': 750, 'cpus': 96000, 'mem': 750*256},
        'shared': {'nodes': 250, 'cpus': 32000, 'mem': 250*256},
        'wholenode': {'nodes': 750, 'cpus': 96000, 'mem': 750*256},
        'wide': {'nodes': 750, 'cpus': 96000, 'mem': 750*256},
        'gpu': {'nodes': 16, 'cpus': 2048, 'mem': 16*512},
        'highmem': {'nodes': 32, 'cpus': 4096, 'mem': 32*1031},
        'azure': {'nodes': 8, 'cpus': 16, 'mem': 8*7}
    }

    for key in par_dict:
        trees.append(IntervalTree())

    for job_idx in tqdm(range(np_array.shape[0])):
        par = np_array[job_idx, idx_dict["PARTITION"]]
        if np_array[job_idx, idx_dict["START_TIME"]] < np_array[job_idx, idx_dict["END_TIME"]]:
            trees[par_dict[par]][np_array[job_idx, idx_dict["START_TIME"]]:np_array[job_idx, idx_dict["END_TIME"]]] = job_idx

    for job_idx in tqdm(range(np_array.shape[0])):
        par_running_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        par = np_array[job_idx, idx_dict["PARTITION"]]
        elig = np_array[job_idx, idx_dict["ELIGIBLE"]]
        remaining_nodes = max_per_par[par]['nodes']
        remaining_cpus = max_per_par[par]['cpus']
        remaining_mem = max_per_par[par]['mem']

        for overlapping in trees[par_dict[par]][elig]:
            if overlapping[2] != job_idx:
                remaining_nodes -= np_array[overlapping[2], idx_dict["REQ_NODES"]]
                remaining_cpus -= np_array[overlapping[2], idx_dict["REQ_CPUS"]]
                remaining_mem -= np_array[overlapping[2], idx_dict["REQ_MEM"]]

        par_running_features[job_idx, 1] = remaining_nodes
        par_running_features[job_idx, 2] = remaining_cpus
        par_running_features[job_idx, 3] = remaining_mem

    new_df = pd.DataFrame(
        {"job_id": par_running_features[:, 0].astype(np.uint32),
         "par_nodes_available": par_running_features[:, 1].astype(np.int32),
         "par_cpus_available": par_running_features[:, 2].astype(np.int32),
         "par_memory_available": par_running_features[:, 3]})

    result = pd.merge(all_df, new_df, how="left", on=["job_id"])
    engine = read_db.create_engine()
    result.to_sql('six_six', engine, if_exists='replace', index=False)
    engine.dispose()

def calculate_remaining_higher_priority_queue_features():
    engine = read_db.create_engine()
    # Read in dataframe
    all_df = pd.read_sql_query("SELECT * FROM six_six ORDER BY eligible", engine)
    df = pd.read_sql_query(
        "SELECT job_id, eligible, start_time, end_time, req_cpus, req_mem, req_nodes, time_limit_raw, partition, priority, par_nodes_available, par_cpus_available, par_memory_available FROM six_six ORDER BY eligible",
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
        "PARTITION": 8,
        "PRIORITY": 9,
        "PAR_NODES_AVAIL": 10,
        "PAR_CPUS_AVAIL": 11,
        "PAR_MEM_AVAIL": 12,
    }

    # Columns: job_id, nodes, cpus, memory
    par_running_queue_features = np.zeros((np_array.shape[0], 4))


    trees = []
    par_dict = {
        'standard': 0,
        'shared': 1,
        'wholenode': 2,
        'wide': 3,
        'gpu': 4,
        'highmem': 5,
        'azure': 6
    }

    for key in par_dict:
        trees.append(IntervalTree())

    for job_idx in tqdm(range(np_array.shape[0])):
        par = np_array[job_idx, idx_dict["PARTITION"]]
        if np_array[job_idx, idx_dict["ELIGIBLE"]] != np_array[job_idx, idx_dict["START_TIME"]]:
            trees[par_dict[par]][np_array[job_idx, idx_dict["ELIGIBLE"]]:np_array[job_idx, idx_dict["START_TIME"]]] = job_idx

    for job_idx in tqdm(range(np_array.shape[0])):
        par = np_array[job_idx, idx_dict["PARTITION"]]
        elig = np_array[job_idx, idx_dict["ELIGIBLE"]]
        par_running_queue_features[job_idx, 0] = np_array[job_idx, idx_dict["JOB_ID"]]
        remaining_nodes = np_array[job_idx, idx_dict["PAR_NODES_AVAIL"]]
        remaining_cpus = np_array[job_idx, idx_dict["PAR_CPUS_AVAIL"]]
        remaining_mem = np_array[job_idx, idx_dict["PAR_MEM_AVAIL"]]

        for overlapping in trees[par_dict[par]][elig]:
            if overlapping[2] != job_idx:
                if np_array[overlapping[2], idx_dict["PRIORITY"]] > np_array[job_idx, idx_dict["PRIORITY"]]:
                        remaining_nodes -= np_array[overlapping[2], idx_dict["REQ_NODES"]]
                        remaining_cpus -= np_array[overlapping[2], idx_dict["REQ_CPUS"]]
                        remaining_mem -= np_array[overlapping[2], idx_dict["REQ_MEM"]]

        par_running_queue_features[job_idx, 1] = remaining_nodes
        par_running_queue_features[job_idx, 2] = remaining_cpus
        par_running_queue_features[job_idx, 3] = remaining_mem

    new_df = pd.DataFrame(
        {"job_id": par_running_queue_features[:, 0].astype(np.uint32),
         "par_nodes_available_running_queue_priority": par_running_queue_features[:, 1].astype(np.int32),
         "par_cpus_available_running_queue_priority": par_running_queue_features[:, 2].astype(np.int32),
         "par_memory_available_running_queue_priority": par_running_queue_features[:, 3]})

    result = pd.merge(all_df, new_df, how="left", on=["job_id"])
    engine = read_db.create_engine()
    result.to_sql('new_features_1milly', engine, if_exists='replace', index=False)
    engine.dispose()




if __name__ == '__main__':
    # calculate_running_features()
    # calculate_queue_features()
    # calculate_higher_priority_queue_features()
    # calculate_user_features()
    calculate_remaining_running_by_partition()
    calculate_remaining_higher_priority_queue_features()

