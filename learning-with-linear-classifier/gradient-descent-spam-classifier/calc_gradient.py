import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sigmoid_function as sf

'''
First, verify that the gradient of the log-loss with respect to the weight vector is:
βππΏπΏ(π,π²,π°,π)βπ°=βπ=1πβπ¦ππ(βπ¦π(π°β€π±π+π))π±π
βππΏπΏ(π,π²,π°,π)βπ=βπ=1πβπ¦ππ(βπ¦π(π°β€π±π+π))
Implement the function gradient, which returns the first derivative with respect to w, b given X, y, w, b. You have access to sigmoid_grader function that returns  π(π§) .

Recall how we got the expressions for the gradient:
βππΏπΏβπ°=ββ[βππ=1logπ(π¦π(π°β€π±π+π))]βπ°=ββπ=1πβ[logπ(π¦π(π°β€π±π+π))]βπ°=ββπ=1π1π(π¦π(π°β€π±π+π))β[π(π¦π(π°β€π±π+π))]βπ°=ββπ=1ππβ²(π¦π(π°β€π±π+π))π(π¦π(π°β€π±π+π))βπ¦ππ±π.
Since we know that  πβ²(π§)=π(π§)(1βπ(π§))  and that  1βπ(π§)=π(βπ§) , our expression finally becomes:
βππΏπΏβπ°=ββπ=1ππ(βπ¦π(π°β€π±π+π))βπ¦ππ±π.
You can similarly derive the gradient w.r.t.  π  as an exercise.

def gradient(X, y, w, b):

'''


def gradient(X, y, w, b):
    '''
    Calculates the gradients of NLL w.r.t. w and b and returns (w_grad, bgrad).

    Input:
        X: data matrix of shape nxd
        y: n-dimensional vector of labels (+1 or -1)
        w: d-dimensional weight vector
        b: scalar bias term

    Output:
        wgrad: d-dimensional vector (gradient vector of w)
        bgrad: a scalar (gradient of b)
    '''
    n, d = X.shape
    wgrad = np.zeros(d)
    bgrad = 0.0

    # YOUR CODE HERE
    temp = np.dot(X, w) + b
    sig = sf.sigmoid(-1 * y * temp)
    sig = sig.reshape(n, 1)
    #     print(sig)
    y = y.reshape(n, 1)
    #     print(y)
    #     wgrad = np.sum(np.dot(np.dot(X.transpose(),sig),y.transpose()*(-1)), axis = 1)
    wgrad = np.sum(np.dot((-y * sig).transpose(), X), axis=0)
    bgrad = np.sum(-y * sig)
    return wgrad, bgrad

