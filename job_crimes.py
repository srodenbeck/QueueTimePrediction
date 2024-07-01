import read_db
import pandas as pd

df = read_db.read_to_df(table="every_job", read_all=True, order_by="eligible", condense_same_times=False)
print("Database read\n\n")

df['elapsed_mins'] = df['elapsed'] / 60
df['wasted_wall_time'] = df['time_limit_raw'] - df['elapsed_mins']
df['percent_wall_time_used'] = df['elapsed_mins'] / df['time_limit_raw']

result = df.groupby('account').agg(
    n_jobs=('time_limit_raw', 'count'),
    average_requested_wall_time=('time_limit_raw', 'mean'),
    average_used_wall_time=('elapsed_mins', 'mean'),
    average_wasted_wall_time=('wasted_wall_time', 'mean'),
    average_wasted_time_percent=('percent_wall_time_used', 'mean'),
    total_wasted_wall_time=('wasted_wall_time', 'sum'),
    most_used_partitions=('partition', pd.Series.mode)
).reset_index()

result['average_requested_wall_time'] = result['average_requested_wall_time'] / 60
result['average_used_wall_time'] = result['average_used_wall_time'] / 60
result['average_wasted_wall_time'] = result['average_wasted_wall_time'] / 60
result['total_wasted_wall_time'] = result['total_wasted_wall_time'] / 60
result['average_wasted_time_percent'] = 100 - (result['average_wasted_time_percent'] * 100)

print(f"FROM {df['eligible'].min()} TO {df['eligible'].max()}")
print("ALL UNITS IN HOURS")
print("TOP 20 HIGHEST WASTED AVERAGE WALL TIME")
result = result.sort_values(by=["average_wasted_wall_time"], ascending=False)
print(result.head(20))

print("TOP 20 HIGHEST WASTED TOTAL WALLTIME")
result = result.sort_values(by=["total_wasted_wall_time"], ascending=False)
print(result.head(20))

result = result.sort_values(by=["average_wasted_wall_time"], ascending=False)

result.to_csv('job_crimes.csv', index=False)