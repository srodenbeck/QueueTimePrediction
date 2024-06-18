# -*- coding: utf-8 -*-

import sklearn.preprocessing as preprocessing
import scipy.stats as stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
import pandas as pd
import read_db


accountUsageDict = dict()

with open('normUsageDict.csv', 'r') as file:
    accountUsageDict['root'] = 1.0
    reader = csv.reader(file, delimiter='|')
    for line in list(reader)[1:]:
        accountUsageDict[line[0]] = float(line[1])


def normalize(train_data, test_data):
    """
    normalize

    Parameters
    ----------
    train_data : ARRAY
        Array of training data. Used to fit the standard scaler.
    test_data : ARRAY
        Array of testing data.

    Returns
    -------
    ARRAY, ARRAY
        Returns two arrays, each standardized according to train_data.

    """
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)


def standardize(train_data, test_data):
    """
    standardize

    Parameters
    ----------
    train_data : ARRAY
        Array of training data. Used to fit the standard scaler.
    test_data : ARRAY
        Array of testing data.

    Returns
    -------
    ARRAY, ARRAY
        Returns two arrays, each standardized according to train_data.

    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)


def memory_to_gigabytes(memory_str):
    """
    memory_to_gigabytes
    
    Parameters
    ----------
    memory_str : STRING
        String in format xxxxU where xxxx is a number and U is a character
        representing the unit of memory used.

    Raises
    ------
    ValueError
        Unit of measurement is not recognized.

    Returns
    -------
    FLOAT
        Returns a float representing the memory of memory_str converted to
        gigabytes.

    """
    if memory_str.endswith('T'):
        return float(memory_str[:-1]) * 1024
    elif memory_str.endswith('G'):
        return float(memory_str[:-1])
    elif memory_str.endswith('M'):
        return float(memory_str[:-1]) / 1024
    elif memory_str == "0":
        return float(0)
    else:
        raise ValueError(f"Unknown memory unit in {memory_str}")


def time_to_seconds(time_str):
    """
    time_to_seconds
    
    Parameters
    ----------
    time_str : STRING
        String in the format D-H:M:S or H:M:S where D represents time in
        days, H represents time in hours, M represents time in minutes, and
        S represents time in seconds.

    Returns
    -------
    INT
        Returns the time in seconds of time_str.

    """
    if '-' in time_str:
        d_hms = time_str.split('-')
        days = int(d_hms[0])
        h, m, s = map(int, d_hms[1].split(':'))
    else:
        days = 0
        h, m, s = map(int, time_str.split(':'))
    return days * 86400 + h * 3600 + m * 60 + s


def boxcox(train_data, test_data):
    """
    Fits a Box-Cox transformation to train_data and then applies it to 
    train_data and test_data.

    Parameters
    ----------
    train_data : ARRAY TYPE
        Array of data which will serve to fit and be applied by the 
        Box-Cox transformation.
    test_data : ARRAY TYPE
        Array of data which will have the Box-Cox transformation applied to
        it with the lambda value resulting from train_data.

    Returns
    -------
    train_data_transformed : ARRAY TYPE
        Transformed version of train_data.
    test_data_transformed : ARRAY TYPE
        Transformed version of test_data.

    """
    train_data_transformed, fit_lambda = stats.boxcox(train_data)
    test_data_transformed = stats.boxcox(test_data, lmbda=fit_lambda)
    return train_data_transformed, test_data_transformed, fit_lambda


def inverse_boxcox(data, lmbda):
    """
    Wrapper for scipy.stats.inv_boxcox() function

    Parameters
    ----------
    data : ARRAY OR FLOAT TYPE
        Number or set of numbers to be converted back to their pre-transformed
        value.
    lmbda : FLOAT
        Float ranging from -5.0 to 5.0 as a result of applying the Box-Cox
        transformation.

    Returns
    -------
    ARRAY OR FLOAT TYPE
        Positive number or set of numbers as a result of reversing the transformation
        for the given lambda.

    """
    return inv_boxcox(data, lmbda)


def scale_min_max(X_train, X_test):
    """
    Applys a min max scaler to X_train and X_test fit to X_trian

    Parameters
    ----------
    X_train : ARRAY TYPE
        Array of floats.
    X_test : ARRAY TYPE
        Array of floats.

    Returns
    -------
    X_train : ARRAY TYPE
        Array of floats scaled between 0 and 1.
    X_test : ARRAY TYPE
        Array of floats roughly scaled between 0 and 1.

    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def scale_min_max_test(X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    return X_test


def scale_log(X_train, X_test):
    # if min(X_train) == 0:
    X_train += 1
    # if min(X_test) == 0:
    X_test += 1
    return np.log(X_train), np.log(X_test)


def scale_log_test(X_test):
    # if min(X_test) == 0:
    X_test += 1
    return np.log(X_test)

    

def accountToNormUsage(account: str):
    """
    accountToNormUsage
    
    Should be used with df apply
    
    Ex: df.apply(accountToNormUsage)

    Parameters
    ----------
    account : str
        String representation of account.

    Returns
    -------
    Normalized value representing the proprotion of resources used by said account.

    """
    if account in accountUsageDict.keys():
        return accountUsageDict[account]
    return 0.0

def make_one_hot(df, col_name):
    """
    make_one_hot
    
    Converts the values within the column matching col_name to onehot encoded values,
    drops the original column, and appends the new one hot columns to the
    dataframe before returning it.

    Parameters
    ----------
    df : DATAFRAME
        Dataframe containing column matching col_name.
    col_name : STRING
        String representing the name of the column to convert to one hot encoding.

    Returns
    -------
    df : DATAFRAME
        Updated dataframe containing one hot encoded values.

    """
    one_hot = pd.get_dummies(df[col_name], drop_first=True)
    df = df.join(one_hot)
    df = df.replace({True: 1, False: 0})
    return df
    
    
    




