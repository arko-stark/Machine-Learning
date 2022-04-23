'''
Implement logistic_regression to train your classifier using
gradient descent with learning rate alpha for max_iter iterations.
You have access to gradient(X, y, w, b) and log_loss_grader(X, y, w, b)
functions. gradient returns wgrad, bgrad and log_loss_grader returns nll scalar.
Please use a constant learning rate alpha throughout (i.e. do not decrease the learning rate).

The idea here is to iteratively update w and b:


'''

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import calc_gradient as cg
import log_loss as ll


def logistic_regression(X, y, max_iter, alpha):
    '''
    Trains the logistic regression classifier on data X and labels y using gradient descent for max_iter iterations with learning rate alpha.
    Returns the weight vector, bias term, and losses at each iteration AFTER updating the weight vector and bias.

    Input:
        X: data matrix of shape nxd
        y: n-dimensional vector of data labels (+1 or -1)
        max_iter: number of iterations of gradient descent to perform
        alpha: learning rate for each gradient descent step

    Output:
        w, b, losses
        w: d-dimensional weight vector
        b: scalar bias term
        losses: max_iter-dimensional vector containing negative log likelihood values AFTER a gradient descent in each iteration
    '''
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = np.zeros(max_iter)

    for step in range(max_iter):
        wgrad, bgrad = cg.gradient(X, y, w, b)
        w = w - alpha * wgrad
        b = b - alpha * bgrad
        losses[step] = ll.log_loss(X, y, w, b)
    return w, b, losses