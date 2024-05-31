import numpy as np
import psycopg2
import pandas as pd
import sqlalchemy
from psycopg2.extensions import register_adapter, AsIs
import config

register_adapter(np.int64, AsIs)

def strip_post(node_str):
    if 'G' in node_str:
        return int(float(node_str[:-1]))
    else:
        return int(float(node_str))

def sqlalc(df, db_config):
    engine = sqlalchemy.create_engine(
        f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
    df.to_sql('jobs', engine, if_exists='append', index=False)

def transform_df(df):
    df = df[["JobID", "UID", "Account", "State", "Partition", "TimelimitRaw", "Submit", "Eligible", "PlannedCPURAW", "Start",
             "End",
             "Priority", "ReqCPUS", "ReqMem", "ReqNodes"]]

    df = df.rename(columns={"JobID": "job_id", "UID": "user_id", "Account": "account", "State": "state",
                            "Partition": "partition",
                            "TimelimitRaw": "time_limit_raw", "Submit": "submit",
                            "Eligible": "eligible", "PlannedCPURAW": "planned_cpu_raw", "Start": "start_time", "End": "end_time",
                            "Priority": "priority", "ReqCPUS": "req_cpus",
                            "ReqMem": "req_mem", "ReqNodes": "req_nodes"})
    df['job_id'] = pd.to_numeric(df['job_id'], errors='coerce')
    df = df.dropna(subset=['job_id'])
    df['time_limit_raw'] = pd.to_numeric(df['time_limit_raw'], errors='coerce')
    df = df.dropna(subset=['time_limit_raw'])
    df.loc[df.start_time == "Unknown", "start_time"] = None
    df.loc[df.end_time == "Unknown", "end_time"] = None
    df['state'] = df['state'].str.partition(' ')[0]

    df['req_mem'] = df['req_mem'].str[:-1]
    df['req_mem'] = pd.to_numeric(df['req_mem'], errors='coerce')
    df = df.dropna(subset=['req_mem'])

    # TODO: Using --units=G appears to break req_nodes column
    df['req_nodes'] = df['req_nodes'].apply(strip_post)

    partition_enum = ['standard', 'shared', 'wholenode', 'wide', 'gpu', 'highmem', 'azure']
    df = df[df['partition'].isin(partition_enum)]

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
                planned_cpu_raw VARCHAR(255),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                priority INTEGER,
                req_cpus INTEGER,
                req_mem REAL,
                req_nodes INTEGER
                )"""
    with conn.cursor() as cursor: cursor.execute(command)

def initialize_db(db_config):
    with psycopg2.connect(dbname=db_config["dbname"], user=db_config["user"], password=db_config["password"], host=db_config["host"],
                          port=db_config["port"]) as conn:
        print("Connected to database")
        create_enum(conn)
        create_table(conn)


if __name__ == "__main__":
    csv_path = "/home/austin/until_2024-05-01.csv"
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
