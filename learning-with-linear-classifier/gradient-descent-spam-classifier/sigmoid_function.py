import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

'''
In sigmoid, implement the sigmoid function:  𝜎(𝑧)=1/(1+𝑒^-z).

'''
def sigmoid(z):
    '''
    Calculates the sigmoid of z.

    Input:
        z: scalar or array of dimension n

    Output:
        scalar or array of dimension n
    '''
    # YOUR CODE HERE
    sigma = 1 / (1 + np.exp(-z))
    return sigma




## testing the code above
def test_sigmoid1():
    h = np.random.rand(10) # input is an 10-dimensional array
    sgmd1 = sigmoid(h)
    return sgmd1.shape == h.shape # output should be a 10-dimensional array

def test_sigmoid2():
    h = np.random.rand(10) # input is an 10-dimensional array
    sgmd1 = sigmoid(h) # compute the sigmoids with your function
    sgmd2 = sigmoid_grader(h) # compute the sigmoids with ground truth funcdtion
    return (np.linalg.norm(sgmd1 - sgmd2) < 1e-5) # check if they agree

def test_sigmoid3():
    x = np.random.rand(1) # input is a scalar
    sgmd1 = sigmoid(x) # compute the sigmoids with your function
    sgmd2 = sigmoid_grader(x) # compute the sigmoids with ground truth function
    return (np.linalg.norm(sgmd1 - sgmd2) < 1e-5) # check if they agree

def test_sigmoid4():
    x = np.array([-1e10,1e10,0]) # three input points: very negative, very positive, and 0
    sgmds = sigmoid(x) # compute the sigmoids with your function
    truth = np.array([0,1,0.5]) # the truth should be 0, 1, 0.5 exactly
    return (np.linalg.norm(sgmds - truth) < 1e-8) # test if this is true