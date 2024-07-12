# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import read_db
import pandas as pd
import seaborn as sns
import numpy as np



features=['time_limit_raw', 'priority', 'req_cpus', 'req_mem', 'jobs_ahead_queue', 'cpus_ahead_queue', 'memory_ahead_queue', 'nodes_ahead_queue', 'time_limit_ahead_queue', 'jobs_running', 'cpus_running', 'memory_running', 'nodes_running', 'time_limit_running', 'par_jobs_ahead_queue', 'par_cpus_ahead_queue', 'par_memory_ahead_queue', 'par_nodes_ahead_queue', 'par_time_limit_ahead_queue', 'par_jobs_running', 'par_cpus_running', 'par_memory_running', 'par_nodes_running', 'par_time_limit_running', 'day_of_week', 'day_of_year']

for i in [11, 23,  9, 19,  0,  5, 15,  3,  1, 13]:
    print(features[i])

df = read_db.read_to_df("jobs_all_2")

df['planned'] = np.log1p(df['planned'])

plt.figure(figsize=(10, 6))
sns.kdeplot(df['planned'], fill=True)
plt.xlabel('Log Transformed Queue Time')
plt.ylabel('Density')
plt.title('Density Plot of Queue Time')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(df['planned'])
plt.xlabel('Log Transformed Queue Time')
plt.ylabel('ECDF')
plt.title('Cumulative Distribution Function (CDF) of Queue Time')
plt.show()