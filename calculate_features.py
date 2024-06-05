import numpy as np
import pandas as pd
import sqlalchemy
import time
import config
from tqdm import tqdm
from intervaltree import IntervalTree, Interval

def create_engine():
    db_config = {
        "dbname": "sacctdata",
        "user": "postgres",
        "password": config.postgres_password,
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
    df = pd.read_sql_query("SELECT job_id, eligible, start_time, end_time FROM jobs ORDER BY eligible LIMIT 200000", engine)
    tree = IntervalTree()
    # df = df[df.start_time != df.end_time]
    df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['end_time'] = df['end_time'].apply(lambda x: x.timestamp()).astype('int64')
    df['eligible'] = df['eligible'].apply(lambda x: x.timestamp()).astype('int64')
    np_array = df.to_numpy()
    np_array = np.append(np_array, np.int32(np.zeros([len(np_array), 1])), 1)

    jobs_running = np.zeros(4096)

    JOB_ID = 0
    ELIGIBLE = 1
    START_TIME = 2
    END_TIME = 3

    max_jobs = -1
    start = time.time()
    for job in tqdm(range(np_array.shape[0])):
        if np_array[job, START_TIME] < np_array[job, END_TIME]:
            tree[np_array[job, START_TIME]:np_array[job, END_TIME]] = np_array[job, 0]

    for job in tqdm(range(np_array.shape[0])):
        jobs_count = 0
        jobs_running[:] = 0
        for idx, overlapping in enumerate(tree[np_array[job, ELIGIBLE]]):
            if overlapping[2] != np_array[job, JOB_ID]:
                jobs_running[idx] = overlapping[2]
                jobs_count += 1

    end = time.time()
    print(end - start)
