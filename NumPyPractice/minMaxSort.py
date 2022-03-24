import numpy as np
b = np.arange(12).reshape(3,4)
print(b)
# remember axis = 0 columnwise , axis = 1 row wise

# max and min in the array
max = np.amax(b)
min = np.amin(b)
print(max, min)

#sorting the array
b1 = np.array([[2,5,1],[5,2,8]])
print(b1)
c = np.sort(b1)
print(c)

# summation
sum1 = b.sum(axis = 0) # sum of each column
sum2 = b.sum(axis = 1) # sum of each column

print (sum1)
print(sum2)