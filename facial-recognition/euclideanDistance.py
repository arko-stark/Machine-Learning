import numpy as np


def innerproduct(X, Z=None):
    if Z is None:  # case when there is only one input (X)
        Z = X;
    G = np.inner(X, Z)
    return G


def calculate_S(X, n, m):
    assert n == X.shape[0]

    S = np.sum(np.square(X), axis=1).reshape(n, 1).repeat(m, axis=1)
    return S


def calculate_R(Z, n, m):
    assert m == Z.shape[0]

    R = np.sum(np.square(Z), axis=1).reshape(1, m).repeat(n, axis=0)
    return R

# final function to return Distance
def l2distance(X, Z=None):
    if Z is None:
        Z = X;

    n, d1 = X.shape
    m, d2 = Z.shape
    assert (d1 == d2), "Dimensions of input vectors must match!"

    # YOUR CODE HERE
    G = innerproduct(X, Z)
    S = calculate_S(X, n, m)
    R = calculate_R(Z, n, m)
    D2 = S + R - 2 * G
    D2[D2 < 0] = 0  # replace non negative values with 0
    return np.sqrt(D2)

# x = np.array([[1,2],[3,4]])
# z = np.array([[1,4],[2,5],[3,6]])
# print(l2distance(x,z))