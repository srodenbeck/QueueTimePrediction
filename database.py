import numpy as np
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import sqlalchemy
from psycopg2.extensions import register_adapter, AsIs

import config

register_adapter(np.int64, AsIs)

with psycopg2.connect(dbname="jobs_db", user="postgres", password=config.postgres_password, host="0.0.0.0", port="5432") as conn:
    print("Connected to database")

    command = """
            CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            account VARCHAR(255),
            state VARCHAR(255),
            partition VARCHAR(255),
            time_limit_raw INTEGER,
            submit TIMESTAMP,
            eligible TIMESTAMP,
            planned VARCHAR(255),
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            priority INTEGER,
            req_cpus INTEGER,
            req_mem VARCHAR(255),
            req_nodes INTEGER
            )"""
    with conn.cursor() as cursor:
        cursor.execute(command)

    df = pd.read_csv("/home/austin/10min.csv", delimiter="|")
    print(df)

    df = df[["JobID", "UID", "Account", "Partition", "TimelimitRaw", "Submit", "Eligible", "Planned", "Start", "End", "Priority", "ReqCPUS", "ReqMem", "ReqNodes"]]

    df.rename(columns={"JobID": "job_id", "UID": "user_id", "Account": "account", "Partition": "partition", "TimelimitRaw": "time_limit_raw", "Submit": "submit",
                       "Eligible": "eligible", "Planned": "planned", "Start": "start_time", "End": "end_time", "Priority": "priority", "ReqCPUS": "req_cpus",
                       "ReqMem": "req_mem", "ReqNodes": "req_nodes"}, inplace=True)


    df.loc[df.start_time == "Unknown", "start_time"] = "1970-01-01"
    # df.start_time = pd.to_datetime(df.start_time)
    df.loc[df.end_time == "Unknown", "end_time"] = "1970-01-01"
    # df.start_time.replace("None", "1970-01-01", inplace=True)
    df = df.fillna("1970-01-01")

    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    query = "INSERT INTO %s(%s) VALUES %%s" % ('jobs', cols)
    with conn.cursor() as cursor:
        extras.execute_values(cursor, query, tuples)
        conn.commit()

