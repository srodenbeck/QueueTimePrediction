#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:40:00 2024

@author: philipwisniewski
"""

# use boxcox(x) to transform it
from scipy.stats import boxcox
# use inv_boxcox(y, lambda) to detransform it
from scipy.special import inv_boxcox
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
import config_file
import numpy as np



def create_engine():
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




if __name__ == "__main__":
    engine = create_engine()
    df = pd.read_sql_query("SELECT req_cpus, eligible, planned FROM new_jobs_odd ORDER BY eligible", engine)

    raw_data = df['req_cpus'].values + 1
    
    bc_results = boxcox(raw_data)
    
    n_bins = 20
    
    figs, axes = plt.subplots(3, 1, figsize=(15,10))
    
    # Raw data
    axes[0].hist(raw_data, n_bins)
    axes[0].set_title("Histogram of Req CPUS")
    axes[0].set_xlabel("Req CPUS")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0)
    
    # Log transformed
    axes[1].hist(np.log(raw_data), n_bins)
    axes[1].set_title("Histogram of Log Tansformed Req CPUS")
    axes[1].set_xlabel("Log Transformed Req CPUS")
    axes[1].set_ylabel("Count")
    axes[1].set_xlim(0)

    # Box-Cox transformed
    tdata = bc_results[0]
    axes[2].hist(tdata, n_bins)
    axes[2].set_title("Histogram of Boxcox Transformed Req CPUS")
    axes[2].set_xlabel("Boxcox Transformed Req CPUS")
    axes[2].set_ylabel("Count")
    axes[2].set_xlim(0)
    
    # Display graphs
    plt.tight_layout()
    plt.show()
    
    
    