import pandas as pd
import sys
import time

def time_to_seconds(time_str):
    if '-' in time_str:
        d_hms = time_str.split('-')
        days = int(d_hms[0])
        h, m, s = map(int, d_hms[1].split(':'))
    else:
        days = 0
        h, m, s = map(int, time_str.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s

def memory_to_gigabytes(memory_str):
    if memory_str.endswith('T'):
        return float(memory_str[:-1]) * 1024
    elif memory_str.endswith('G'):
        return float(memory_str[:-1])
    elif memory_str.endswith('M'):
        return float(memory_str[:-1]) / 1024
    else:
        raise ValueError(f"Unknown memory unit in {memory_str}")

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

df = pd.read_csv("all_data.csv", delimiter="|")
df = df[df['State'] == "COMPLETED"]
df = df[['JobID', 'Start', 'Eligible', 'End', 'Priority', 'ReqCPUS', 'ReqMem', 'ReqNodes', 'TimelimitRaw', 'QOSRAW', 'Partition', 'TimelimitRaw', 'Planned']]

df['PlannedRaw'] = df['Planned'].apply(time_to_seconds)
df['ReqMem'] = df['ReqMem'].apply(memory_to_gigabytes)
df['Partition'] = df['Partition'].map(partition_dict)

print(df['Partition'].unique())

for col in df.columns:
    # print(col, "has a max value of", df[col].max(), "and a min value of", df[col].min())
    print(df[col].head())
    print()


df['Start'] = pd.to_datetime(df['Start'])
df['Eligible'] = pd.to_datetime(df['Eligible'])
df['End'] = pd.to_datetime(df['Eligible'])



print("Starting to make jobs _____ at Eligible")
start = time.time()
# Making list of jobids where the job is in the queue upon submitting
df['Jobs in Queue at Eligible'] = [[] for _ in range(len(df))]
df['Jobs Running at Eligible'] = [[] for _ in range(len(df))]

# Assumes that data is in chronological order. Otherwise would take much more time
# (think O(n^2) rather than O(n))

# Arbitrary value for patience for how many jobs ahead to look without success
patience = 50

df.reset_index(inplace=True)
# TODO: FIX SO THAT IT GOES TO PREVIOUS VALUE EVEN THOUGH SOME INDICES ARE TAKEN OUT
df = df.sort_values(by=['Eligible'])
for row in range(len(df)):
    offset = 0
    temper = 0
    # TODO: Potentially add another while loop going the other direction
    # goes until temper exceeds patience. If succesful, temper is reset to 0.
    while temper < patience and row - offset >= 0:
        offset += 1
        comparison = df.loc[row, 'Eligible']
        if df.loc[row - offset, 'End'] <= comparison:
            temper += 1
            continue
        if df[row - offset]['Start'] <= comparison:
            temper = 0
            df.loc[row, 'Jobs Running at Eligible'].append(df.loc[row - offset, 'JobID'])
        elif df[row - offset]['Eligible'] <= comparison:
            temper = 0
            df.loc[row, 'Jobs in Queue at Eligible'][row].append(df.loc[row - offset, 'JobID'])
        else:
            print("ERROR: Issue in eligible sort")
            sys.exit(-1)

end = time.time()
print(f"Finished. Code section took {end - start} seconds")

