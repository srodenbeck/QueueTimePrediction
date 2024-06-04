import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

np.random.seed(0)

def time_to_seconds(time_str):
    """
    Function to convert time_str which is in format D-HH-MM-SS to seconds
    """
    if '-' in time_str:
        d_hms = time_str.split('-')
        days = int(d_hms[0])
        h, m, s = map(int, d_hms[1].split(':'))
    else:
        days = 0
        h, m, s = map(int, time_str.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s

def memory_to_gigabytes(memory_str):
    """
    Function to convert memory_str which is in format ____U where U is unit
    into an integer representing size in gigabytes
    """
    if memory_str.endswith('T'):
        return float(memory_str[:-1]) * 1024
    elif memory_str.endswith('G'):
        return float(memory_str[:-1])
    elif memory_str.endswith('M'):
        return float(memory_str[:-1]) / 1024
    else:
        raise ValueError(f"Unknown memory unit in {memory_str}")

# Dictionary to convert partition into enum
partition_dict = {
    "standard": 1,
    "shared": 2,
    "debug": 3,
    "gpu": 4,
    "highmem": 5,
    "gpu-debug": 6,
    "wide": 7,
    "benchmarking": 8,
    "wholenode": 9,
    "azure": 10
}

# Read file
df = pd.read_csv("all_data.csv", delimiter="|")
# Potentially remove
df = df[df['State'] == 'COMPLETED']

# Apply transforms to make data into integers/standardized format
df['PlannedRaw'] = df['Planned'].apply(time_to_seconds)
df['ReqMem'] = df['ReqMem'].apply(memory_to_gigabytes)
df['Partition'] = df['Partition'].map(partition_dict)

# Make strings into datetime format
df['Start'] = pd.to_datetime(df['Start'])
df['Eligible'] = pd.to_datetime(df['Eligible'])
df['End'] = pd.to_datetime(df['Eligible'])

# Replace unlimited
df['TimelimitRaw'] = df['TimelimitRaw'].replace('UNLIMITED', 40000).astype(int)

# Potential feature
df['CPUTime'] = df['TimelimitRaw'] * df['ReqCPUS']

# print(df.columns)
# Independent variables
x_var_names = ['Priority', 'ReqCPUS', 'QOSRAW', 'Partition', 'TimelimitRaw', 'ReqMem', 'ReqNodes', 'CPUTime']
# Dependent variable
target = 'PlannedRaw'

# Path to where images will be stored
path_root = "Graphs/TrendLines/"
path_end = "_minute_plus.png"

# y = df['PlannedRaw']

# Only show jobs with wait time greater than a minute
df = df[df['PlannedRaw'] > 60]

# Make graph for each independent variable
for name in x_var_names:
    print(name)
    plt.figure(figsize=(10,6))
    f, ax = plt.subplots()
    df = df.sort_values(by=name)
    # moved above
    # df_copy = df[df['PlannedRaw'] > 60]

    # Index can be adjusted to only graph first X% of data
    # index = int(0.99 * len(df))
    index = len(df)
    X = df[name].iloc[:index]
    y = df['PlannedRaw'].iloc[:index]

    plt.scatter(X, y)
    plt.xlabel(name)
    plt.ylabel('Planned (s)')
    plt.title(f"{name} vs Planned (Wait longer than 1 minute)")
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, y)
    plt.plot(X, slope*X + intercept, "r-")
    ax.text(0.1, 0.9, f"R2 value: {r_value * r_value}", transform=ax.transAxes)
    # plt.annotate("r-squared = {:.3f}".format(r_value), (0,1))
    # plt.annotate("p-value = {:.5f}".format(p_value), (0,0.9))
    plt.savefig(path_root + name + path_end)
    plt.show()
    

