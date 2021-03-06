**Part 1**: 
``Implement findknn [Graded]
Implement the function findknn, which should find the  ๐  nearest neighbors ( ๐โค๐ ) of a set of vectors within a given training data set. With xTr of size  ๐ร๐  and xTe of size  ๐ร๐ , the call of:

[I, D] = findknn(xTr, xTe, k)
should result in two matrices I and D, both of dimensions  ๐ร๐ , where  ๐  is the number of input vectors in xTe. The matrix I[i, j] is the index of the  ๐๐กโ  nearest neighbor of the vector xTe[j, :].

So, for example, if we set i = I(1, 3), then xTr[i, :] is the first nearest neighbor of vector xTe[3, :]. The second matrix D returns the corresponding distances. So D[i, j] is the distance of xTe[j, :] to its  ๐๐กโ  nearest neighbor.

l2distance(X, Z) from the last exercise is readily available to you with the following specification:

"""
Computes the Euclidean distance matrix.
Syntax: D = l2distance(X, Z)
Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d    
Output:
    Matrix D of size nxm
        D(i, j) is the Euclidean distance of X(i, :) and Z(j, :)
call with only one input: l2distance(X) = l2distance(X, X).
"""
One way to use l2distance() is as follows:

Compute distances D between xTr and xTe using l2distance.
Get indices of k-smallest distances for each testing point to create the I matrix.
Use I to re-order D or create D by getting the k-smallest distances for each testing point.
You may find np.argsort(D, axis=0) and np.sort(D, axis=0) useful when implementing findknn.