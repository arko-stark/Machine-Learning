import numpy as np

# initialize an array with zeroes
x1 = np.zeros((2,3))
print(x1)

#initialize the array with ones
x2 = np.ones((2,3,4))
print(x2)

# empty array with random values
x3 = np.empty((2,3))
print(x3)

# create an identity matrix
x4 = np.eye(3)
print(x4)


# return a sequence of number in an array
x5 = np.arange(2,10)
print(x5)

# if we want number of elements instead of step
x6 = np.linspace(0,2,9) # we want 9 equally spaced elements between 0 and 2
print(x6)