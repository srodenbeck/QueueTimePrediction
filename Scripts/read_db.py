import numpy as np
import pandas as pd
import sqlalchemy
import time
import config

start = time.time()


db_config = {
        "dbname": "sacctdata",
        "user": "postgres",
        "password": config.postgres_password,
        "host": "slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
        "port": "5432"
}

engine = sqlalchemy.create_engine(
        f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
df = pd.read_sql_query("SELECT * FROM jobs", engine)
end = time.time()
print(f"time = {end - start}")
print(df.head(n=10))


arr = df.to_numpy()
end = time.time()
print(arr)
print(f"time = {end - start}")
