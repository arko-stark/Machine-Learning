import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(file='heart_disease_train.csv', label=True):
    """
    Returns the data matrix and optionally the corresponding label vector.

    Input:
        file: filename of the dataset
        label: a boolean to decide whether to return the labels or not

    Output:
        X: (numpy array) nxd data matrix of patient attributes
        y: (numpy array) n-dimensional vector of labels (if label=False, y is not returned)
    """
    X = None
    y = None

    # YOUR CODE HERE
    # referring to the documentation here : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    n = pd.read_csv(file)
    #     Xt = n.iloc[:, :-1]
    #     yt = n.iloc[:, -1]
    #     X = Xt.to_numpy()
    #     y = yt.to_numpy()
    if label == True:
        Xt = n.iloc[:, :-1]
        yt = n.iloc[:, -1]
        X = Xt.to_numpy()
        y = yt.to_numpy()
        return X, y
    elif label == False:
        return n.to_numpy()
    else:
        pass