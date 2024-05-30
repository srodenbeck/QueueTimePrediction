import numpy as np
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import sqlalchemy
from psycopg2.extensions import register_adapter, AsIs

import config

register_adapter(np.int64, AsIs)


def sqlalc(df):
    db_username = 'postgres'
    db_password = config.postgres_password
    db_host = 'slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu'
    db_port = '5432'
    db_name = 'sacctdata'
    engine = sqlalchemy.create_engine(
        f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
    df.to_sql('jobs', engine, if_exists='append', index=False)

def transform_df(df):
    df = df[["JobID", "UID", "Account", "State", "Partition", "TimelimitRaw", "Submit", "Eligible", "Planned", "Start",
             "End",
             "Priority", "ReqCPUS", "ReqMem", "ReqNodes"]]

    df = df.rename(columns={"JobID": "job_id", "UID": "user_id", "Account": "account", "State": "state",
                            "Partition": "partition",
                            "TimelimitRaw": "time_limit_raw", "Submit": "submit",
                            "Eligible": "eligible", "Planned": "planned", "Start": "start_time", "End": "end_time",
                            "Priority": "priority", "ReqCPUS": "req_cpus",
                            "ReqMem": "req_mem", "ReqNodes": "req_nodes"})
    df.loc[df.start_time == "Unknown", "start_time"] = None
    df.loc[df.end_time == "Unknown", "end_time"] = None
    df['req_mem'] = df['req_mem'].str[:-1]
    df['state'] = df['state'].str.partition(' ')[0]
    df = df.fillna(pd.NA)
    return df


def create_enum(conn):
    command = """DROP TYPE IF EXISTS partition_enum CASCADE"""
    with conn.cursor() as cursor: cursor.execute(command)
    command = """DROP TYPE IF EXISTS state_enum CASCADE"""
    with conn.cursor() as cursor: cursor.execute(command)
    command = """CREATE TYPE partition_enum AS ENUM ('standard', 'shared', 'wholenode', 'wide', 'gpu', 'highmem', 
        'debug', 'gpu-debug')"""
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
                planned VARCHAR(255),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                priority INTEGER,
                req_cpus INTEGER,
                req_mem REAL,
                req_nodes INTEGER
                )"""
    with conn.cursor() as cursor: cursor.execute(command)


with psycopg2.connect(dbname="sacctdata", user="postgres", password=config.postgres_password, host="slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
                      port="5432") as conn:
    print("Connected to database")
    create_enum(conn)
    create_table(conn)

csv_path = "/home/austin/10min.csv"
df = pd.read_csv(csv_path, delimiter="|")
df = transform_df(df)

sqlalc(df)
