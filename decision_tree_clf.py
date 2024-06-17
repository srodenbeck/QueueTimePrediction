#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:37:11 2024

@author: philipwisniewski
"""

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


# Convert data to classes
intervals = {
    "less than 5 minutes": 5 * 60,
    "less than 15 minutes": 15 * 60,
    "less than an hour": 60 * 60,
    "less than 4 hours": 4 * 60 * 60,
    "less than a day": 24 * 60 * 60,
    "more than a day": float('inf')  # Representing values more than a day
}
classes = {
    "less than 5 minutes": 0,
    "less than 15 minutes": 1,
    "less than an hour": 2,
    "less than 4 hours": 3,
    "less than a day": 4,
    "more than a day": 5
}
NUM_CLASSES = len(intervals)

def categorize_time(queue_time):
    for label, interval in intervals.items():
        if queue_time < interval:
            return label
    return "more than a day"


# Read in data
df = pd.read_csv("/Users/philipwisniewski/spyder/reu-p4/QueueTimePrediction/data_with_running.csv")
df = df.iloc[8192:,:].reset_index()
print(df.head())

# Split data (no validation but oh well)
training = df.iloc[:90000, :]
testing = df.iloc[90000:, :]
X_train = training[['req_cpus', 'req_mem', 'req_nodes', 'time_limit_raw',
                   'jobs_running', 'cpus_running', 'nodes_running',
                   'memory_running']].values
y_train = training['planned'].values
X_test = testing[['req_cpus', 'req_mem', 'req_nodes', 'time_limit_raw',
                   'jobs_running', 'cpus_running', 'nodes_running',
                   'memory_running']].values
y_test = testing['planned'].values

y_train_categorical = [categorize_time(queue_time) for queue_time in y_train]
y_test_categorical = [categorize_time(queue_time) for queue_time in y_test]

y_train_encoded = [classes[x] for x in y_train_categorical]
y_test_encoded = [classes[x] for x in y_test_categorical]

# Make and train model
clf = DecisionTreeClassifier(random_state=0, max_depth=5)
clf.fit(X_train, y_train_encoded)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test_encoded, y_pred))

# Print out model architecture
plot_tree(clf, feature_names=['req_cpus', 'req_mem', 'req_nodes', 'time_limit_raw',
                   'jobs_running', 'cpus_running', 'nodes_running',
                   'memory_running'])
plt.savefig('out.pdf')
plt.show()
