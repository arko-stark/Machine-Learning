def classify_linear(xs, w, b=None):
    """
    function preds=classify_linear(xs,w,b)

    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)

    Output:
    preds: predictions (1xn)
    """
    w = w.flatten()
    predictions = np.zeros(xs.shape[0])

    # YOUR CODE HERE
    n, d = xs.shape
    if b == None:
        b = 0
    for i in range(0, n):
        if (np.dot(xs[i], w) + b) >= 0:
            predictions[i] = +1
        else:
            predictions[i] = -1
    return predictions

# Run this self-test to check that your linear classifier correctly classifies the points in a linearly separable dataset

def test_linear1():
    xs = np.random.rand(50000,20)-0.5 # draw random data
    xs = np.hstack([ xs, np.ones((50000,1)) ])
    w0 = np.random.rand(20+1)
    w0[-1] = -0.1 # with bias -0.1
    ys = classify_linear(xs,w0)
    uniquepredictions=np.unique(ys) # check if predictions are only -1 or 1
    return set(uniLinearquepredictions)==set([-1,1])

def test_linear2():
    xs = np.random.rand(1000,2)-0.5 # draw random data
    xs = np.hstack([ xs, np.ones((1000, 1)) ])
    w0 = np.array([0.5,-0.3,-0.1]) # define a random hyperplane with bias -0.1
    ys = np.sign(xs.dot(w0)) # assign labels according to this hyperplane (so you know it is linearly separable)
    return (all(np.sign(ys*classify_linear(xs,w0))==1.0))  # the original hyperplane (w0,b0) should classify all correctly

runtest(test_linear1, 'test_linear1')
runtest(test_linear2, 'test_linear2')