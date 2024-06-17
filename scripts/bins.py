import numpy as np
import pandas as pd
import read_db

print("PERCENT OF JOBS PLANNED/QUEUE TIME FALLING WITHIN VARIOUS BINS")

def print_hist(jobs):
    df = read_db.read_to_df(table="new_jobs_all", read_all=False, jobs=jobs)
    df.planned = df.planned / 60
    bins = ["0-1min", "1-10min", "10-60min", "1-4hr", "4-24hr", "1-7day", "7-30day", "other"]
    bin_nums = [0, 1, 10, 60, 60*4, 60*24, 60*24*7, 60*24*30, 10000000000]
    np_hist = np.histogram(df['planned'], bins=bin_nums)[0]
    np_hist = np_hist / np_hist.sum()
    np_hist = np_hist * 100

    first_data = df.submit.min()
    last_data = df.submit.max()
    print("--------------------------------------------------------------")
    print(f"{first_data} - {last_data}")
    for i, bin in enumerate(bins):
        print(f"{bin}: {np_hist[i]:.2f}%")
    print("--------------------------------------------------------------\n")

print_hist(10000)
print_hist(100000)
print_hist(1000000)
print_hist(500000000)
