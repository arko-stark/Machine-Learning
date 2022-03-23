import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(x)
# number of dimensions/axes in the array
axis = x.ndim
print(axis)

# dimensions of the array
dim1 = x.shape
print(dim1)

# size of the array : total number of elements
s1 = x.size
print(s1)

# data type of the array
print(x.dtype)
