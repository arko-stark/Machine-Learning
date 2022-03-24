
import numpy as np
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(a)
print(b)
c = a-b
print(c)

# matrix Multiplication
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
print(A.dot(B))