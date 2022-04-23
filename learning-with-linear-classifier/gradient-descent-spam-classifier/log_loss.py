import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

'''
Now you will compute the negative log likelihood in log_loss. 
You are given the label vector y and the data matrix X with n data points as row vectors. 
The negative log likelihood (𝑁𝐿𝐿) is defined as follows:

𝑁𝐿𝐿=−log𝑃(𝐲|𝐗;𝐰,𝑏)=−log∏𝑖=1𝑛𝑃(𝑦𝑖|𝐱𝑖;𝐰,𝑏)=−∑𝑖=1𝑛log𝑃(𝑦𝑖|𝐱𝑖;𝐰,𝑏)=−∑𝑖=1𝑛log𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏)).
While we only computed the probability of a positive label in y_pred, 
now we will account for the actual 𝑦 value. You can use your implementation of y_pred for 
log_loss or reimplement to account for the 𝑦 -- the latter yields cleaner code.


'''


def log_loss(X, y, w, b=0):
    '''
    Calculates the negative log likelihood for dataset (X, y) using the weight vector w and bias b.

    Input:
        X: data matrix of shape nxd
        y: n-dimensional vector of labels (+1 or -1)
        w: d-dimensional vector
        b: scalar (optional, default is 0)

    Output:
        scalar
    '''
    assert np.sum(np.abs(y)) == len(y)  # check if all labels in y are either +1 or -1

    # YOUR CODE HERE
    preds = y_pred(X, w, b)
    nll_t = np.sum((y + 1) * np.log(preds) - (y - 1) * np.log(1 - preds))
    nll = -nll_t / 2
    return nll