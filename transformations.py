# -*- coding: utf-8 -*-

import sklearn.preprocessing as preprocessing
import scipy.stats as stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import MinMaxScaler


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
    