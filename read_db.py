import numpy as np
import pandas as pd
import sqlalchemy
import time

import config_file


def create_engine():
    """
            create_engine()

            Connects a sqlalchemy engine to a postgres database

            Returns
            -------
            engine : SQLALCHEMY ENGINE
                Returns an instance of a sqlalchemy engine to access postgres database.

        """
    db_config = {
        "dbname": "sacctdata",
        "user": "postgres",
        "password": config_file.postgres_password,
        "host": "slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu",
        "port": "5432"
    }

    engine = sqlalchemy.create_engine(
        f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}')
    return engine


def read_to_df(table, read_all=True, jobs=10000, order_by="submit"):
    """
       read_to_np()

       Reads in data from the table jobs to a dataframe and converts
       it to a numpy array.

       Parameters
       ----------
       engine : SQLALCHEMY ENGINE
           Instance of sqlalchemy engine to access postgres database.

       Returns
       -------
       np_array : NUMPY ARRAY
           Returns an array containing the data from the jobs table
           in postgres database.

    """
    engine = create_engine()
    if order_by == "submit":
        if read_all:
            df = pd.read_sql_query(f"SELECT * FROM {table} WHERE submit <= '2024-04-18'", engine)
        else:
            df = pd.read_sql_query(f"SELECT * FROM {table} WHERE submit <= '2024-04-18' ORDER BY submit DESC LIMIT {jobs}", engine)
    elif order_by == "random":
        if read_all:
            df = pd.read_sql_query(f"SELECT * FROM {table} WHERE submit <= '2024-04-18' ORDER BY random()", engine)
        else:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE submit <= '2024-04-18' ORDER BY random() DESC LIMIT {jobs}", engine)
    return df
