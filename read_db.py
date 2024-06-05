import numpy as np
import pandas as pd
import sqlalchemy
import time
import config

def read_to_df():
        db_config = {
                "dbname": "sacctdata",
                "user": "postgres",
                "password": config.postgres_password,
                "host": "slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
                "port": "5432"
        }

        engine = sqlalchemy.create_engine(
                f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
        df = pd.read_sql_query("SELECT * FROM jobs ORDER BY RANDOM() LIMIT 10000", engine)
        return df
