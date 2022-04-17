import sys
import numpy as np
import matplotlib.pyplot as plt
print('You\'re running python %s' % sys.version.split(' ')[0])


## Generate Random Data
# First, let's generate some data. We will sample  𝑥  randomly between  0  and  1 ,
# and let  𝑦  be sampled as:  𝑦=3𝑥+4+𝜖 , where  𝜖∼𝑁(0,0.01)  is noise sampled from a Gaussian random variable
# of mean  0  and standard deviation  0.01 .

N = 40
X = np.random.rand(N, 1) # Sample N points randomly along X-axis
X = np.hstack((X, np.ones((N, 1))))  # Add a constant dimension
w = np.array([3, 4]) # defining a linear function
y = X @ w + (np.random.randn(N) * 0.1) # defining labels

# Visualize Data
plt.plot(X[:, 0], y, '.')
plt.show()


#Learning using closed for solution
"""
𝐰=(𝐗𝑇𝐗)−1𝐗𝑇𝐲
"""

w_closed = np.linalg.inv(X.T @ X) @ (X.T @ y)

# alternately to the line above you can use "w_closed_np = np.linalg.solve(X.T @ X, X.T @ y)"

# visualize the closed form solution
def plot_linear(X, y, w):
    plt.plot(X[:, 0], y, '.') # plot the points
    z = np.array([[0,1],      # define two points with X-value 0 and 1 (and constant dimension)
                  [1,1]])
    plt.plot(z[:, 0], z @ w, 'r') # draw line w_closed through these two points
    plt.show()
plot_linear(X, y, w_closed)