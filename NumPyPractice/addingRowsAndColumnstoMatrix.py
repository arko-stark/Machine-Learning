import numpy as np

a= np.array([[1,2,3,4],[7,8,9,10]])
b= np.array([[4,5,6,7],[1,2,3,4]])

print(np.vstack((a,b))) # vertical stacking concatenates vectors along the first axis

print(np.hstack((a,b))) # horizontal stacking concatenates vectors along the second axis


# Find the maximum elements in A along the first axis (axis = 0)
# and add it to the sum of elements in B along the first axis.

amax = np.max(a,axis = 0)
print(amax)
sumb = np.sum(b,axis = 0)
print(amax+sumb)