# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../account_count.csv")

df = df.iloc[:100, :]

print(df.head())

df['cumulative_jobs'] = df['count'].cumsum()

# Calculate the percentage of total jobs submitted
total_jobs = 3808960
df['percentage_of_total'] = (df['cumulative_jobs'] / total_jobs) * 100


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(df) + 1), df['percentage_of_total'], marker='o')
plt.xlabel('Number of Users Included')
plt.ylabel('Percentage of Total Jobs Submitted')
plt.title('Cumulative Percentage of Jobs Submitted by Users')
plt.grid(True)
plt.show()