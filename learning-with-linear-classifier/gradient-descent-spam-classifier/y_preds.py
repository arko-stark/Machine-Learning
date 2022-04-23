import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

'''
Implement the function y_pred(X, w) that computes 𝑃(𝑦𝑖=1|𝐱𝑖;𝐰,𝑏) for each row-vector 𝐱𝑖 in the matrix X.

Recall that:
𝑃(𝑦𝑖|𝐱𝑖;𝐰,𝑏)=𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏))
Since y_pred calculates the probability of the positive label +1, 
𝜎(𝑦𝑖(𝐰⊤𝐱𝑖+𝑏)) simplifies to 𝜎(𝐰⊤𝐱𝑖+𝑏), which you must output.


'''
def y_pred(X, w, b=0):
    '''
    Calculates the probability of the positive class.

    Input:
        X: data matrix of shape nxd
        w: d-dimensional vector
        b: scalar (optional, default is 0)

    Output:
        n-dimensional vector
    '''

    # YOUR CODE HERE
    temp = np.dot(X, w) + b
    ypred = sigmoid(temp)
    return ypred