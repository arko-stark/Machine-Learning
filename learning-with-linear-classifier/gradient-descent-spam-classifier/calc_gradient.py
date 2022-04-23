import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sigmoid_function as sf

'''
First, verify that the gradient of the log-loss with respect to the weight vector is:
∂𝑁𝐿𝐿(𝐗,𝐲,𝐰,𝑏)∂𝐰=∑𝑖=1𝑛−𝑦𝑖𝜎(−𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))𝐱𝑖
∂𝑁𝐿𝐿(𝐗,𝐲,𝐰,𝑏)∂𝑏=∑𝑖=1𝑛−𝑦𝑖𝜎(−𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))
Implement the function gradient, which returns the first derivative with respect to w, b given X, y, w, b. You have access to sigmoid_grader function that returns  𝜎(𝑧) .

Recall how we got the expressions for the gradient:
∂𝑁𝐿𝐿∂𝐰=−∂[∑𝑛𝑖=1log𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))]∂𝐰=−∑𝑖=1𝑛∂[log𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))]∂𝐰=−∑𝑖=1𝑛1𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))∂[𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))]∂𝐰=−∑𝑖=1𝑛𝜎′(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))⋅𝑦𝑖𝐱𝑖.
Since we know that  𝜎′(𝑧)=𝜎(𝑧)(1−𝜎(𝑧))  and that  1−𝜎(𝑧)=𝜎(−𝑧) , our expression finally becomes:
∂𝑁𝐿𝐿∂𝐰=−∑𝑖=1𝑛𝜎(−𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))⋅𝑦𝑖𝐱𝑖.
You can similarly derive the gradient w.r.t.  𝑏  as an exercise.

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

