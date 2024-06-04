import numpy as np
import psycopg2
import pandas as pd
import sqlalchemy
from psycopg2.extensions import register_adapter, AsIs
import config

register_adapter(np.int64, AsIs)


def memory_to_gigabytes(memory_str):
    if memory_str.endswith('T'):
        return float(memory_str[:-1]) * 1024
    elif memory_str.endswith('G'):
        return float(memory_str[:-1])
    elif memory_str.endswith('M'):
        return float(memory_str[:-1]) / 1024
    elif memory_str == "0":
        return float(0)
    else:
        raise ValueError(f"Unknown memory unit in {memory_str}")

def time_to_seconds(time_str):
    if '-' in time_str:
        d_hms = time_str.split('-')
        days = int(d_hms[0])
        h, m, s = map(int, d_hms[1].split(':'))
    else:
        days = 0
        h, m, s = map(int, time_str.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s

def sqlalc(df, db_config):
    engine = sqlalchemy.create_engine(
        f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
    df.to_sql('jobs', engine, if_exists='append', index=False)

def transform_df(df):
    df = df[["JobID", "UID", "Account", "State", "Partition", "TimelimitRaw", "Submit", "Eligible", "Elapsed", "Planned", "Start",
             "End", "Priority", "ReqCPUS", "ReqMem", "ReqNodes", "ReqTRES", "QOS"]]

    df = df.rename(columns={"JobID": "job_id", "UID": "user_id", "Account": "account", "State": "state",
                            "Partition": "partition",
                            "TimelimitRaw": "time_limit_raw", "Submit": "submit",
                            "Eligible": "eligible", "Elapsed": "elapsed", "Planned": "planned", "Start": "start_time", "End": "end_time",
                            "Priority": "priority", "ReqCPUS": "req_cpus",
                            "ReqMem": "req_mem", "ReqNodes": "req_nodes", "ReqTRES": "req_tres", "QOS": "qos"})
    # Remove incomplete jobs.

    df.loc[df.planned == "INVALID", "planned"] = None
    df['job_id'] = pd.to_numeric(df['job_id'], errors='coerce')
    df['time_limit_raw'] = pd.to_numeric(df['time_limit_raw'], errors='coerce')
    df.loc[df.start_time == "Unknown", "start_time"] = None
    df.loc[df.end_time == "Unknown", "end_time"] = None
    df['state'] = df['state'].str.partition(' ')[0]
    df = df.dropna(subset=['job_id'])
    df = df.dropna(subset=['planned'])
    df = df.dropna(subset=['elapsed'])
    df = df.dropna(subset=['time_limit_raw'])
    df = df.dropna(subset=['req_mem'])
    df = df.dropna(subset=['start_time'])

    # Format fields.
    df['req_mem'] = df['req_mem'].apply(memory_to_gigabytes)
    df['planned'] = df['planned'].apply(time_to_seconds)
    df['elapsed'] = df['elapsed'].apply(time_to_seconds)
    partition_enum = ['standard', 'shared', 'wholenode', 'wide', 'gpu', 'highmem', 'azure']
    df = df[df['partition'].isin(partition_enum)]

    # Change Null values for pandas.NA
    df = df.fillna(pd.NA)
    return df


def create_enum(conn):
    command = """DROP TYPE IF EXISTS partition_enum CASCADE"""
    with conn.cursor() as cursor: cursor.execute(command)
    command = """DROP TYPE IF EXISTS state_enum CASCADE"""
    with conn.cursor() as cursor: cursor.execute(command)
    command = """CREATE TYPE partition_enum AS ENUM ('standard', 'shared', 'wholenode', 'wide', 'gpu', 'highmem', 
    'azure')"""
    with conn.cursor() as cursor: cursor.execute(command)
    command = """CREATE TYPE state_enum AS ENUM ('COMPLETED', 'CANCELLED', 'FAILED', 'REQUEUED', 'NODE_FAIL', 'PENDING', 
            'OUT_OF_MEMORY', 'TIMEOUT')"""
    with conn.cursor() as cursor: cursor.execute(command)

def create_table(conn):
    command = """DROP TABLE IF EXISTS jobs"""
    with conn.cursor() as cursor: cursor.execute(command)

    command = """
                CREATE TABLE IF NOT EXISTS jobs (
                job_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                account VARCHAR(255),
                state state_enum,
                partition partition_enum,
                time_limit_raw INTEGER,
                submit TIMESTAMP,
                eligible TIMESTAMP,
                elapsed INTEGER,
                planned INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                priority INTEGER,
                req_cpus INTEGER,
                req_mem REAL,
                req_nodes INTEGER,
                req_tres VARCHAR(255),
                QOS VARCHAR(255)
                )"""
    with conn.cursor() as cursor: cursor.execute(command)

def initialize_db(db_config):
    with psycopg2.connect(dbname=db_config["dbname"], user=db_config["user"], password=db_config["password"], host=db_config["host"],
                          port=db_config["port"]) as conn:
        print("Connected to database")
        create_enum(conn)
        create_table(conn)


if __name__ == "__main__":
    csv_path = "/home/austin/until_2024-05-02.csv"
    db_config = {
        "dbname": "sacctdata",
        "user": "postgres",
        "password": config.postgres_password,
        "host": "slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
        "port": "5432"
    }
    initialize_db(db_config)
    df = pd.read_csv(csv_path, delimiter="|")
    df = transform_df(df)
    sqlalc(df, db_config)
