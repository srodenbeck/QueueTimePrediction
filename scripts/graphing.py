# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import read_db
import numpy as np

df = read_db.read_to_df("jobs_everything_all_2")

elapsed_mins = df['elapsed'] / 60

linear_fit = np.polyfit(elapsed_mins, df['time_limit_raw'], 1)
quadratic_fit = np.polyfit(elapsed_mins, df['time_limit_raw'], 2)

plt.scatter(elapsed_mins, df['time_limit_raw'])
plt.xlabel("Elapsed (m)")
plt.ylabel("Time Limit Raw (m)")
plt.plot(elapsed_mins, linear_fit[0] * elapsed_mins + linear_fit[1], "r-")
plt.scatter(elapsed_mins, quadratic_fit[0] * elapsed_mins * elapsed_mins + quadratic_fit[1] * elapsed_mins + quadratic_fit[2], "g")
plt.show()