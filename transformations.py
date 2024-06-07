# -*- coding: utf-8 -*-

import sklearn.preprocessing as preprocessing


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
