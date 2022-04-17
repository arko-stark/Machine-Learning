import numpy as np
def perceptron_update(x, y, w):
    """
    function w=perceptron_update(x,y,w);

    Updates the perceptron weight vector w using x and y
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions

    Output:
    w : weight vector after updating (d)
    """
    # YOUR CODE HERE
    w = w + np.dot(np.transpose(x), y)
    return w

# test of the perceptron_update function
# This self test will check that your perceptron_update
# function returns the correct values for input vector [0,1], label -1, and weight vector [1,1]

def test_perceptron_update1():
    x = np.array([0,1])
    y = -1
    w = np.array([1,1])
    w1 = perceptron_update(x,y,w)
    return (w1.reshape(-1,) == np.array([1,0])).all()

def test_perceptron_update2():
    x = np.random.rand(25)
    y = 1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1-x)<1e-8


def test_perceptron_update3():
    x = np.random.rand(25)
    y = -1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1+x)<1e-8


runtest(test_perceptron_update1, 'test_perceptron_update1')
runtest(test_perceptron_update2, 'test_perceptron_update2')
runtest(test_perceptron_update3, 'test_perceptron_update3')