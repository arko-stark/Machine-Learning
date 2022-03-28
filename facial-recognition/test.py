import numpy as np
x = np.array([[1,4],[2,5],[1,6],[9,1],[5,6]]) # example data
k = 2 # return the indices of the 2 smallest values
I = np.argsort(x, axis=0)# by row
print(I)
print('\n'*5)
I = np.argsort(x, axis=0)[0:k+1,:]# by column
print(I)
print('\n'*5)
D = np.sort(x, axis=0)[0:k+1,:]
print(D)
np.fla
