import numpy as np
import pandas as pd
import sqlalchemy
import time
import config_file

def read_to_df(read_all=True, jobs=10000):
        db_config = {
                "dbname": "sacctdata",
                "user": "postgres",
                "password": config_file.postgres_password,
                "host": "slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
                "port": "5432"
        }

        engine = sqlalchemy.create_engine(
                f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
        if read_all:
                df = pd.read_sql_query("SELECT * FROM jobs", engine)
        else:
                df = pd.read_sql_query(f"SELECT * FROM jobs ORDER BY submit DESC LIMIT {jobs}", engine)
        return df
