#!/usr/bin/env python
# coding: utf-8

# <h2>About this Project</h2>
# <p>In this project, you will implement a kernelized SVM. You will generate linearly separable and non-linearly separable datasets, write kernel, loss, and gradient functions for SVMs that support a variety of different kernels, and then visualize the decision boundary created.</p>
# 
# <h3>Evaluation</h3>
# 
# <p><strong>This project must be successfully completed and submitted in order to receive credit for this course. Your score on this project will be included in your final grade calculation.</strong><p>
#     
# <p>You are expected to write code where you see <em># YOUR CODE HERE</em> within the cells of this notebook. Not all cells will be graded; code input cells followed by cells marked with <em>#Autograder test cell</em> will be graded. Upon submitting your work, the code you write at these designated positions will be assessed using an "autograder" that will run all test cells to assess your code. You will receive feedback from the autograder that will identify any errors in your code. Use this feedback to improve your code if you need to resubmit. Be sure not to change the names of any provided functions, classes, or variables within the existing code cells, as this will interfere with the autograder. Also, remember to execute all code cells sequentially, not just those you’ve edited, to ensure your code runs properly.</p>
#     
# <p>You can resubmit your work as many times as necessary before the submission deadline. If you experience difficulty or have questions about this exercise, use the Q&A discussion board to engage with your peers or seek assistance from the instructor.<p>
# 
# <p>Before starting your work, please review <a href="https://s3.amazonaws.com/ecornell/global/eCornellPlagiarismPolicy.pdf">eCornell's policy regarding plagiarism</a> (the presentation of someone else's work as your own without source credit).</p>
# 
# <h3>Submit Code for Autograder Feedback</h3>
# 
# <p>Once you have completed your work on this notebook, you will submit your code for autograder review. Follow these steps:</p>
# 
# <ol>
#   <li><strong>Save your notebook.</strong></li>
#   <li><strong>Mark as Completed —</strong> In the blue menu bar along the top of this code exercise window, you’ll see a menu item called <strong>Education</strong>. In the <strong>Education</strong> menu, click <strong>Mark as Completed</strong> to submit your code for autograder/instructor review. This process will take a moment and a progress bar will show you the status of your submission.</li>
# 	<li><strong>Review your results —</strong> Once your work is marked as complete, the results of the autograder will automatically be presented in a new tab within the code exercise window. You can click on the assessment name in this feedback window to see more details regarding specific feedback/errors in your code submission.</li>
#   <li><strong>Repeat, if necessary —</strong> The Jupyter notebook will always remain accessible in the first tabbed window of the exercise. To reattempt the work, you will first need to click <strong>Mark as Uncompleted</strong> in the <strong>Education</strong> menu and then proceed to make edits to the notebook. Once you are ready to resubmit, follow steps one through three. You can repeat this procedure as many times as necessary.</li>
# <p>You can also download a copy of this notebook in multiple formats using the <strong>Download as</strong> option in the <strong>File</strong> menu above.</p>

# <h2>Getting Started</h2>
# <h3>Python Initialization</h3> 
# 
# Please run the following code to initialize your Python kernel. You should be running a version of Python 3.x. </p>

# In[1]:


import numpy as np
from helper import *
import matplotlib.pyplot as plt
import sys

print('You\'re running python %s' % sys.version.split(' ')[0])


# ### Generate and Visualize Data
# 
# Before we start, let's generate some data and visualize the training set. We are going to use the linearly separable data that we used in our previous project!

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
xTr,yTr = generate_data()
visualize_2D(xTr, yTr)


# ## Kernel SVM
# 
# In this assignment, you need to implement three functions:
# 1. `computeK` that computes the kernel function efficiently,
# 2. `loss` that calculates the kernelized version of the unconstrained squared hinge loss and the regularizer, and
# 3. `grad` that calculates the gradients of the loss with respect to the kernel SVM's model parameters.

# ### Part One: Compute K [Graded]
# 
# In **`computeK`**, calculate the values of different kernel functions given inputs `X` and `Z`. You will return a matrix $\mathsf{K}$ such that the entry $\mathsf{K}_{ij} = \mathsf{K} \left( \mathbf{x}_i, \mathbf{z}_j \right)$ where $\mathsf{K} \left( \mathbf{x}_i, \mathbf{z}_j \right) = \phi \left( \mathbf{x}_i \right)^\top \phi \left(\mathbf{z}_j \right)$. As you have seen so far, computing $\phi \left(\mathbf{x}_i \right)$ and $\phi \left(\mathbf{z}_j \right)$ explicitly and taking the dot product can be computationally expensive. Consequently, we will use the dot product expression $\mathsf{K} \left( \mathbf{x}_i, \mathbf{z}_j \right)$ without ever computing $\phi \left(\mathbf{x}_i \right)$ and $ \phi \left(\mathbf{z}_j \right)$.
# 
# `computeK` takes in the parameter `kerneltype` to decide which of the three different kernel functions to calculate:
# - `kerneltype == 'linear'`: $\mathsf{K} (\mathbf{X}, \mathbf{Z}) = \mathbf{X}^\top \mathbf{Z}$
# - `kerneltype == 'polynomial'`: $\mathsf{K} (\mathbf{X}, \mathbf{Z}) = \left( 1 + \mathbf{X}^\top \mathbf{Z} \right)^{p}$ where `kpar = p`
# - `kerneltype == 'rbf'`: $\mathsf{K} (\mathbf{X}, \mathbf{Z}) = e^{-\frac{||\mathbf{X}-\mathbf{Z}||^2}{\sigma^2}}$ where `kpar` = $\frac{1}{\sigma^2}$
# 
# **Implementation Notes:**
# - When calculating the RBF kernel, you can use the `l2distance(X, Z)` function that we have provided to you. It calculates the pairwise L2 distance $||\mathbf{X} - \mathbf{Z}||$ efficiently.
# - [`np.power(a, p)`](https://numpy.org/doc/stable/reference/generated/numpy.power.html) raises all entries of the vector or matrix `a` to the power `p`.

# In[3]:


def computeK(kerneltype, X, Z, kpar=1):
    """
    Computes a matrix K such that K[i, j] = K(x_i, z_j). The kernel operation is defined by kerneltype with parameter kpar.

    Input:
        kerneltype: either of ['linear', 'polynomial', 'rbf']
        X: nxd data matrix
        Z: mxd data matrix
        kpar: kernel parameter (inverse sigma^2 in case of 'rbf', degree p in case of 'polynomial')

    Output:
        K : nxm kernel matrix
    """
    assert kerneltype in ['linear', 'polynomial', 'rbf'], 'Kernel type %s not known.' % kerneltype
    assert X.shape[1] == Z.shape[1], 'Input dimensions do not match'

    K = None

    # YOUR CODE HERE
    # Conditional operator for 3 different Kernel types
    if kerneltype == 'linear':
        K = np.dot(X,np.transpose(Z))
    elif kerneltype == 'polynomial':
        K = np.power((1+np.dot(X,np.transpose(Z))),kpar)
    elif kerneltype == 'rbf':
        K = np.exp(-kpar*np.square(l2distance(X,Z)))
    else :
        print('Wrong type kernel')
    return K
    
    


# In[4]:


# These tests test whether your computeK() is implemented correctly

xTr_test, yTr_test = generate_data(100)
xTr_test2, yTr_test2 = generate_data(50)
n, d = xTr_test.shape

# Checks whether computeK compute the kernel matrix with the right dimension
def computeK_test1():
    s1 = (computeK('rbf', xTr_test, xTr_test2, kpar=1).shape == (100, 50))
    s2 = (computeK('polynomial', xTr_test, xTr_test2, kpar=1).shape == (100, 50))
    s3 = (computeK('linear', xTr_test, xTr_test2, kpar=1).shape == (100, 50))
    return (s1 and s2 and s3)

# Checks whether the kernel matrix is symmetric
def computeK_test2():
    k_rbf = computeK('rbf', xTr_test, xTr_test, kpar=1)
    s1 = np.allclose(k_rbf, k_rbf.T)
    k_poly = computeK('polynomial', xTr_test, xTr_test, kpar=1)
    s2 = np.allclose(k_poly, k_poly.T)
    k_linear = computeK('linear', xTr_test, xTr_test, kpar=1)
    s3 = np.allclose(k_linear, k_linear.T)
    return (s1 and s2 and s3)

# Checks whether the kernel matrix is positive semi-definite
def computeK_test3():
    k_rbf = computeK('rbf', xTr_test2, xTr_test2, kpar=1)
    eigen_rbf = np.linalg.eigvals(k_rbf)
    eigen_rbf[np.isclose(eigen_rbf, 0)] = 0
    s1 = np.all(eigen_rbf >= 0)
    k_poly = computeK('polynomial', xTr_test2, xTr_test2, kpar=1)
    eigen_poly = np.linalg.eigvals(k_poly)
    eigen_poly[np.isclose(eigen_poly, 0)] = 0
    s2 = np.all(eigen_poly >= 0)
    k_linear = computeK('linear', xTr_test2, xTr_test2, kpar=1)
    eigen_linear = np.linalg.eigvals(k_linear)
    eigen_linear[np.isclose(eigen_linear, 0)] = 0
    s3 = np.all(eigen_linear >= 0)
    return (s1 and s2 and s3)

# Checks whether computeK compute the right kernel matrix with rbf kernel
def computeK_test4():
    k = computeK('rbf', xTr_test, xTr_test2, kpar=1)
    k2 = computeK_grader('rbf', xTr_test, xTr_test2, kpar=1)
    
    return np.linalg.norm(k - k2) < 1e-5

# Checks whether computeK compute the right kernel matrix with polynomial kernel
def computeK_test5():
    k = computeK('polynomial', xTr_test, xTr_test2, kpar=1)
    k2 = computeK_grader('polynomial', xTr_test, xTr_test2, kpar=1)
    
    return np.linalg.norm(k - k2) < 1e-5

# Checks whether computeK compute the right kernel matrix with linear kernel
def computeK_test6():
    k = computeK('linear', xTr_test, xTr_test2, kpar=1)
    k2 = computeK_grader('linear', xTr_test, xTr_test2, kpar=1)
    
    return np.linalg.norm(k - k2) < 1e-5


runtest(computeK_test1, 'computeK_test1')
runtest(computeK_test2, 'computeK_test2')
runtest(computeK_test3, 'computeK_test3')
runtest(computeK_test4, 'computeK_test4')
runtest(computeK_test5, 'computeK_test5')
runtest(computeK_test6, 'computeK_test6')


# In[5]:


# Autograder test cell - worth 1 point
# runs computeK_test1


# In[6]:


# Autograder test cell - worth 1 point
# runs computeK_test2


# In[7]:


# Autograder test cell - worth 1 point
# runs computeK_test3


# In[8]:


# Autograder test cell - worth 1 point
# runs computeK_test4


# In[9]:


# Autograder test cell - worth 1 point
# runs computeK_test5


# In[10]:


# Autograder test cell - worth 1 point
# runs computeK_test5


# Previously in linear SVM, we could pass in $\mathbf{w}$ to calculate the unconstrained square hinge loss and the $l_2$-regularizer. However, for the kernelized version of the loss function, $\mathbf{w}$ is written as a linear combination of the $n$ training examples $\phi \left( \mathbf{x}_1 \right), \dots, \phi \left( \mathbf{x}_n \right)$ and thus we will need to pass in the coefficients in the linear combination and training points.
# 
# As we will see in the following section, after substituting $\mathbf{w}$ with a linear combination of training examples, we can simplify the kernel SVM loss function quite a bit.
# 
# ### Kernelized SVM Loss Function
# 
# We assume a training set of $n$ examples. Recall how $\mathbf{w}$ could be written as a linear combination of the training samples:
# $$
#     \mathbf{w} = \sum_{j = 1}^n \alpha_j y_j \phi \left( \mathbf{x}_j \right) = \sum_{j=1}^n \beta_j \phi \left( \mathbf{x}_j \right)
# $$
# where we define new model parameters $\beta_j = \alpha_j y_j$ for simpler expressions going forward.
# 
# Also recall the expression for the regularizer and the unconstrained squared hinge loss over any $m$ points $\phi\left( \mathbf{z}_1 \right), \dots, \phi \left( \mathbf{z}_m \right)$. If we substitute $\mathbf{w}$ into the loss, we get the following expression. **Observe that only the squared hinge loss is dependent on the $m$ points; the regularizer is independent.**
# $$
#     \begin{aligned}
#     \ell(\mathbf{w}, b) &= \underbrace{\mathbf{w}^\top \mathbf{w}}_{l_{2} \text{ regularizer} } +  C \underbrace{ \sum_{i=1}^{m} \max \left[ 1-y_{i} \left( \mathbf{w}^\top \phi \left(\mathbf{z}_i\right) + b \right), 0 \right]^2}_{ \text{squared hinge loss} }\\
#         &= \left( \sum_{j=1}^n \beta_j \phi \left( \mathbf{x}_j \right) \right)^\top \left( \sum_{j=1}^n \beta_j \phi \left( \mathbf{x}_j \right) \right)
#             + C \sum_{i=1}^m \max \left[ 1 - y_i \left( \left( \sum_{j=1}^n \beta_j \phi \left( \mathbf{x}_j \right) \right)^\top \phi \left( \mathbf{z}_i \right) + b \right), 0 \right]^2\\
#         &= \sum_{j = 1}^n \sum_{k = 1}^n \beta_j \beta_k \phi \left( \mathbf{x}_j \right)^\top \phi \left( \mathbf{x}_k \right)
#             + C \sum_{i=1}^{m} \max \left[ 1-y_{i} \left( \sum_{j = 1}^n \beta_j \phi \left(\mathbf{x}_j \right)^\top \phi \left(\mathbf{z}_i \right)+b \right), 0 \right]^2
#     \end{aligned}
# $$
# 
# Let us now replace all the dot product terms $\phi(\mathbf{a})^\top \phi(\mathbf{b})$ with $\mathsf{K}(\mathbf{a}, \mathbf{b})$ for all data points $\mathbf{a}, \mathbf{b}$.
# $$
#     \begin{aligned}
#     \ell\left( \beta_1, \dots, \beta_n, b \right) = \underbrace{\sum_{j = 1}^n \sum_{k = 1}^n \beta_j \beta_k \mathsf{K} \left( \mathbf{x}_j, \mathbf{x}_k \right)}_{l_{2} \text{ regularizer}}
#         + C \underbrace{ \sum_{i=1}^{m} \max \left[ 1-y_{i} \left(\sum_{j = 1}^n \beta_j \mathsf{K} \left(\mathbf{x}_j, \mathbf{z}_i \right)+b \right), 0 \right]^2}_{\text{squared hinge loss}}
#     \end{aligned}
# $$
# 
# Let us pause here to observe a few facts:
# - When $\mathbf{w}$ is written as a linear combination of the $n$ training points, the loss is then optimized over the parameters $\beta_1, \dots, \beta_n, b$.
# - The $l_2$-regularizer is a dot-product of $\mathbf{w}$ with itself. Since $\mathbf{w}$ is a linear combination of the $n$ training points, the regularizer only depends on the dot-products between $n$ training points.
# - The loss never needs $\phi(\mathbf{x})$ explicitly. It only needs the dot products between data points.
# 
# Now, let us move ahead and simplify the loss function to a vector form. First, we will simplify the $l_{2}$ regularizer. Define $\mathbf{\beta} = \left[\beta_1, \dots, \beta_n \right]^\top$ and $\mathsf{K}_{nn}$ of size $n \times n$ to be the kernel matrix calculated on the training set of $n$ points. Precisely, the entry $\mathsf{K}_{nn}[j, k] = \mathsf{K} \left(\mathbf{x}_j, \mathbf{x}_k \right)$. Thus, the $l_2$-regularizer can be written as the quadratic form:
# $$
#     \sum_{j = 1}^n \sum_{k = 1}^n \beta_j \beta_k \mathsf{K} \left(\mathbf{x}_j, \mathbf{x}_k \right) = \mathbf{\beta}^\top \mathsf{K}_{nn} \mathbf{\beta}
# $$
# 
# Similarly, we can define the kernel matrix $\mathsf{K}_{nm}$ of size $n \times m$, the $(j, i)^{th}$ entry is $\mathsf{K}_{nm}[j, i] = \mathsf{K} \left( \mathbf{x}_j, \mathbf{z}_i \right)$. Hence, the summation term in the hinge loss can be expressed as: 
# $$
#     \sum_{j = 1}^n \beta_j \mathsf{K} \left(\mathbf{x}_j, \mathbf{z}_i \right) = \mathbf{\beta}^\top \mathsf{K}_{nm}[:, i]
# $$
# where $\mathsf{K}_{nm}[:, i]$ is the $i^{th}$ column of $\mathsf{K}_{nm}$.
# 
# Combining the two simplifications we have, we arrive at the following final expression for the loss function: 
# $$
#     \begin{aligned}
#     \ell\left( \mathbf{\beta}, b \right) = \underbrace{\mathbf{\beta}^\top \mathsf{K}_{nn} \mathbf{\beta}}_{l_{2} \text{ regularizer}}
#         + C \underbrace{ \sum_{i=1}^{m} \max \left[ 1-y_{i} \left( \mathbf{\beta}^\top \mathsf{K}_{nm}[:, i] + b \right), 0 \right]^2}_{\text{squared hinge loss}}
#     \end{aligned}
# $$
# 
# **During training, we minimize the training loss, with $\mathsf{K}_{nm}$ replaced by $\mathsf{K}_{nn}$, to get the optimal $\mathbf{\beta}, b$. Then we fix $\mathbf{\beta}, b$ and evaluate the loss value on testing points with $\mathsf{K}_{nm}$ in the squared hinge loss.**
# 
# Note that the loss function we have above is very similar to the vanilla linear SVM. The key differences are: 
# 1. Instead of $\mathbf{w}$, we have $\mathbf{\beta}$ to optimize for.
# 2. The $l_{2}$-regularizer $\mathbf{w}^\top \mathbf{w}$ is replaced by $\mathbf{\beta}^\top \mathsf{K}_{nn} \mathbf{\beta}$ to account for using $\mathbf{\beta}$ instead of $\mathbf{w}$.
# 3. The inner product $\mathbf{w}^\top \phi \left( \mathbf{z}_i \right)$ in the hinge loss is changed to $\mathbf{\beta}^T \mathsf{K}_{nm}[:, i]$.
# 
# Since each entry of $\mathsf{K}_{nn}$ and $\mathsf{K}_{nm}$ can be calculated by a simple formula in `computeK`, the kernel SVM is efficiently optimizable even if the $\phi$ function is in a high-dimensional space.
# 
# ### Part Two: Compute Loss [Graded]
# 
# Now you will implement the function **`loss`**. The function takes in model parameters `beta, b`, $n$ training points as `xTr, yTr` and $m$ testing points as `xTe, yTe`, along with hyperparameters `C, kerneltype, kpar`. You will need to calculate both kernel matrices $\mathsf{K}_{nn}$ and $\mathsf{K}_{nm}$ using `computeK` on `xTr, xTr` and `xTr, xTe` respectively.
# 
# When we use the `loss` function later on, we are going to be a little clever: we will use it both for testing and training loss.
# - During training, we will call `loss(beta, b, xTr, yTr, xTr, yTr, C, kerneltype, kpar)` so that the hinge loss gets calculated on $\mathsf{K}_{nn}$.
# - During testing, we will just call `loss(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar)` so that the hinge loss gets calculated on $\mathsf{K}_{nm}$.
# 
# Therefore, you should implement `loss` keeping in mind how we will call it during training and testing.

# In[11]:


def loss(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar=1):
    """
    Calculates the loss (regularizer + squared hinge loss) for testing data against training data and parameters beta, b.
    
    Input:
        beta  : n-dimensional vector that stores the linear combination coefficients
        b     : bias term, a scalar
        xTr   : nxd dimensional data matrix (training set, each row is an input vector)
        yTr   : n-dimensional vector (training labels, each entry is a label)
        xTe   : mxd dimensional matrix (test set, each row is an input vector)
        yTe   : m-dimensional vector (test labels, each entry is a label)
        C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
        kerneltype: either of ['linear', 'polynomial', 'rbf']
        kpar  : kernel parameter (inverse sigma^2 in case of 'rbf', degree p in case of 'polynomial')
    
    Output:
        loss_val : the total loss obtained with (beta, xTr, yTr, b) on xTe and yTe, a scalar
    """
    
    loss_val = 0.0
    # compute the kernel values between xTr and xTr 
    kernel_train = computeK(kerneltype, xTr, xTr, kpar)
    # compute the kernel values between xTr and xTe
    kernel_test = computeK(kerneltype, xTr, xTe, kpar)
    
    # YOUR CODE HERE

    
    # calculate squared hinge loss
    pred = kernel_test@beta + b
    margin = yTe*pred
    sqhl = C*(np.sum(np.maximum(1-margin,0)**2))
    
    # calculate l2 regularizer
    l2 = beta@kernel_train@beta
    
    #calculate loss val
    loss_val = l2+sqhl
    
    return loss_val


# In[12]:


# # test cell please ignore
# xTr, yTr = generate_data()
# n, d = xTr.shape
# beta = np.zeros(n)
# b = np.zeros(1)
# # kpar = 1
# kernel_train = computeK('rbf', xTr, xTr, kpar = 1)
# l2 = np.dot(beta,kernel_train,beta)
# print(l2.shape)
# print(l2)


# In[13]:


# # test cell 2 please ignore
# beta = np.zeros(n)
# b = np.zeros(1)
# # loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
# # print(loss_val)
# loss_val_grader = loss_grader(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
# # print(loss_val_grader)
# loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
# print(loss_val)
# print(np.linalg.norm(loss_val - loss_val_grader))


# In[14]:


# These tests test whether your loss() is implemented correctly

xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape

# Check whether your loss() returns a scalar
def loss_test1():
    beta = np.zeros(n)
    b = np.zeros(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
    
    return np.isscalar(loss_val)


# Check whether your loss() returns a nonnegative scalar
def loss_test2():
    beta = np.random.rand(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
    
    return loss_val >= 0

# Check whether you implement l2-regularizer correctly
def loss_test3():
    beta = np.random.rand(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 0, 'rbf')
    loss_val_grader = loss_grader(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 0, 'rbf')
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

# Check whether you implement square hinge loss correctly
def loss_test4():
    beta = np.zeros(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
    loss_val_grader = loss_grader(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

# Check whether you implement square hinge loss correctly
def loss_test5():
    beta = np.zeros(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
    loss_val_grader = loss_grader(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, 'rbf')
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

# Check whether you implement loss correctly
def loss_test6():
    beta = np.zeros(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 100, 'rbf')
    loss_val_grader = loss_grader(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 100, 'rbf')
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

# Check whether you implement loss correctly for testing data
def loss_test7():
    xTe_test, yTe_test = generate_data()
    m, _ = xTe_test.shape
    
    beta = np.zeros(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTe_test, yTe_test, 100, 'rbf')
    loss_val_grader = loss_grader(beta, b, xTr_test, yTr_test, xTe_test, yTe_test, 100, 'rbf')
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

runtest(loss_test1,'loss_test1')
runtest(loss_test2,'loss_test2')
runtest(loss_test3,'loss_test3')
runtest(loss_test4,'loss_test4')
runtest(loss_test5,'loss_test5')
runtest(loss_test6,'loss_test6')
runtest(loss_test7,'loss_test7')


# In[15]:


# Autograder test cell - worth 1 point
# runs loss_test1


# In[16]:


# Autograder test cell - worth 1 point
# runs loss test2


# In[17]:


# Autograder test cell - worth 1 point
# runs loss test3


# In[18]:


# Autograder test cell - worth 1 point
# runs loss test4


# In[19]:


# Autograder test cell - worth 1 point
# runs loss test5


# In[20]:


# Autograder test cell - worth 1 point
# runs loss test6


# In[21]:


# Autograder test cell - worth 1 point
xTe_test, yTe_test = generate_data()
m, _ = xTe_test.shape

assert loss_test7()
# runs loss test7


# ### Part Three: Compute Gradient [Graded]
# 
# Now, you will implement the function **`grad`** that computes the gradients of the loss function with respect to the parameters, similar to what you did in the Linear SVM project. `grad` outputs the gradient with respect to $\mathbf{\beta}$ (`beta_grad`) and $b$ (`bgrad`). Unlike `loss`, `grad` is only called during the training phase; consequently, the input parameters don't include `xTe, yTe`. Remember that the squared hinge loss is calculated with $\mathsf{K}_{nn}$ when training, and so you would just need to call `computeK` on `xTr, xTr` here.
# 
# The gradients are given by:
# $$
#     \begin{aligned}
#     \frac{\partial \ell}{\partial \mathbf{\beta}} &=  2 \mathsf{K}_{nn} \mathbf{\beta} + C \sum_{i=1}^{n} 2 \max \left[ 1-y_{i} \left(\mathbf{\beta}^\top \mathsf{K}_{nn}[:, i] + b \right), 0 \right] \left( - y_i \mathsf{K}_{nn}[:, i] \right) \mathbf{1}_{1 - y_i \left( \mathbf{\beta}^\top \mathsf{K}_{nn}[:, i] + b \right) > 0}\\
#     \frac{\partial \ell}{\partial b} &=  C \sum_{i=1}^{n} 2 \max \left[ 1-y_{i} \left( \mathbf{\beta}^\top \mathsf{K}_{nn}[:, i] + b \right), 0 \right] \left(-y_i \right) \mathbf{1}_{1 - y_i \left( \mathbf{\beta}^\top \mathsf{K}_{nn}[:, i] + b \right) > 0}
#     \end{aligned}
# $$
# where the indicator function is:
# $$
# \mathbf{1}_{1 - y_i \left( \mathbf{\beta}^\top \mathsf{K}_{nn}[:, i] + b \right) > 0} = \left\{ \begin{array}{ll}1 & \text{if }1 - y_i \left( \mathbf{\beta}^\top \mathsf{K}_{nn}[:, i] + b \right) > 0 \\ 0 & \text{otherwise} \end{array} \right.
# $$

# In[36]:


def grad(beta, b, xTr, yTr, C, kerneltype, kpar=1):
    """
    Calculates the gradients of the loss function with respect to beta and b.
    
    Input:
        beta  : n-dimensional vector that stores the linear combination coefficients
        b     : bias term, a scalar
        xTr   : nxd dimensional data matrix (training set, each row is an input vector)
        yTr   : n-dimensional vector (training labels, each entry is a label)
        C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
        kerneltype: either of ['linear', 'polynomial', 'rbf']
        kpar  : kernel parameter (inverse sigma^2 in case of 'rbf', degree p in case of 'polynomial')
    
    Output:
        beta_grad, bgrad
        beta_grad :  n-dimensional vector (the gradient of loss with respect to the beta)
        bgrad     :  scalar (the gradient of loss with respect to the bias, b)
    """
    
    n, d = xTr.shape
    
    beta_grad = np.zeros(n)
    bgrad = np.zeros(1)
    
    # compute the kernel values between xTr and xTr 
    kernel_train = computeK(kerneltype, xTr, xTr, kpar)
    
    # YOUR CODE HERE
    pred = np.dot(kernel_train,beta)+b
    m = yTr*pred
    sqh = np.maximum(1-m,0)
    # transform true false indicator into 0's and 1's
    ind = ((1-m) >0).astype(int)
    bgrad = C*np.sum(2*sqh*ind*(-yTr),axis = 0)
    beta_grad = 2*(np.dot(kernel_train,beta)) + C*np.sum((2*sqh*ind*-yTr).reshape(-1,1)*kernel_train, axis = 0)
    return beta_grad, bgrad


# In[37]:


# # Test Cell please ignore
# xTr_test, yTr_test = generate_data()
# print(xTr.shape)
# n, d = xTr_test.shape
# kpar = 1
# kerneltype = 'rbf'
# beta = np.random.rand(n)
# b = np.random.rand(1)
# C = 10
# kernel_train = computeK(kerneltype, xTr, xTr, kpar)
# pred = np.dot(kernel_train,beta)+b
# # print(pred)
# # print(pred.shape)
# m = yTr*pred
# # print(m)
# # print(m.shape)
# sqh = np.maximum(1-m,0)
# # transform true false indicator into 0's and 1's
# ind = ((1-m) >0).astype(int)
# # print(ind)
# bgrad = C*np.sum(2*sqh*ind*(-yTr),axis = 0)
# print(bgrad)
# beta_grad = 2*(np.dot(kernel_train,beta)) + C*np.sum((2*sqh*ind*-yTr).reshape(-1,1)*kernel_train, axis = 0)
# print(beta_grad)
# print(len(beta_grad))


# In[38]:


# These tests test whether your grad() is implemented correctly

xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape
    
# Checks whether grad returns a tuple
def grad_test1():
    beta = np.random.rand(n)
    b = np.random.rand(1)
    out = grad(beta, b, xTr_test, yTr_test, 10, 'rbf')
    return len(out) == 2

# Checks the dimension of gradients
def grad_test2():
    beta = np.random.rand(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 10, 'rbf')
    return len(beta_grad) == n and np.isscalar(bgrad)

# Checks the gradient of the l2 regularizer
def grad_test3():
    beta = np.random.rand(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 0, 'rbf')
    beta_grad_grader, bgrad_grader = grad_grader(beta, b, xTr_test, yTr_test, 0, 'rbf')
    return (np.linalg.norm(beta_grad - beta_grad_grader) < 1e-5) and         (np.linalg.norm(bgrad - bgrad_grader) < 1e-5)

# Checks the gradient of the square hinge loss
def grad_test4():
    beta = np.zeros(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 1, 'rbf')
    beta_grad_grader, bgrad_grader = grad_grader(beta, b, xTr_test, yTr_test, 1, 'rbf')
    return (np.linalg.norm(beta_grad - beta_grad_grader) < 1e-5) and         (np.linalg.norm(bgrad - bgrad_grader) < 1e-5)

# Checks the gradient of the loss
def grad_test5():
    beta = np.random.rand(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 10, 'rbf')
    beta_grad_grader, bgrad_grader = grad_grader(beta, b, xTr_test, yTr_test, 10, 'rbf')
    return (np.linalg.norm(beta_grad - beta_grad_grader) < 1e-5) and         (np.linalg.norm(bgrad - bgrad_grader) < 1e-5)

runtest(grad_test1, 'grad_test1')
runtest(grad_test2, 'grad_test2')
runtest(grad_test3, 'grad_test3')
runtest(grad_test4, 'grad_test4')
runtest(grad_test5, 'grad_test5')


# In[39]:


# Autograder test cell - worth 1 point
# runs grad test1


# In[40]:


# Autograder test cell - worth 1 point
# runs grad test2


# In[41]:


# Autograder test cell - worth 1 point
# runs grad test3


# In[42]:


# Autograder test cell - worth 1 point
# runs grad test4


# In[43]:


# Autograder test cell - worth 1 point
# runs grad test5


# ## Test the Kernelized Algorithm

# Using the cell below, you can call the optimization routine that we have implemented for you to see the final loss of your model. The loss will not be 0 since it includes the non-zero regularization term. To check only squared hinge loss term, we can subtract the regularization term from the final loss.

# In[44]:


beta_sol, bias_sol, final_loss = minimize(objective=loss, grad=grad, xTr=xTr, yTr=yTr, C=1000, kerneltype='linear', kpar=1)
print('The Final Loss of your model is: {:0.4f}'.format(final_loss))

K_nn = computeK('linear', xTr, xTr, kpar=1)
reg = beta_sol @ K_nn @ beta_sol
print('The Final Squared Hinge Loss of your model is: {:0.4f}'.format(final_loss - reg))


# If everything is implemented correctly, you should be able to get a training error of zero when you run the following cell.

# In[45]:


svmclassify = lambda x: np.sign(computeK('linear', x, xTr, 1).dot(beta_sol) + bias_sol)

predsTr=svmclassify(xTr)
trainingerr=np.mean(np.sign(predsTr)!=yTr)
print("Training error: %2.4f" % trainingerr)


# <h3>Visualize the Decision Boundary</h3>
# 
# Also, when you visualize the classifier, you should see a max margin separator.

# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
visclassifier(svmclassify, xTr, yTr)


# Let's visualize a different kind of nonlinear data, a spiral dataset.

# In[47]:


xTr_spiral,yTr_spiral,xTe_spiral,yTe_spiral = spiraldata()

get_ipython().run_line_magic('matplotlib', 'inline')
visualize_2D(xTr_spiral, yTr_spiral)


# Since the dataset is nonlinear, we are going to use the RBF kernel.

# In[48]:


beta_sol_spiral, bias_sol_spiral, final_loss_spiral = minimize(objective=loss, grad=grad, xTr=xTr_spiral, yTr=yTr_spiral, C=100, kerneltype='rbf', kpar=1)
print('The Final Loss of your model is: {:0.4f}'.format(final_loss_spiral))

K_nn = computeK('rbf', xTr_spiral, xTr_spiral, kpar=1)
reg = beta_sol_spiral @ K_nn @ beta_sol_spiral
print('The Final Squared Hinge Loss of your model is: {:0.4f}'.format(final_loss_spiral - reg))


# If you do everything correctly, your training error and test error should both be zero!

# In[50]:


svmclassify_spiral = lambda x: np.sign(computeK('rbf', xTr_spiral, x, 1).transpose().dot(beta_sol_spiral) + bias_sol_spiral)

predsTr_spiral = svmclassify_spiral(xTr_spiral)
trainingerr_spiral = np.mean(predsTr_spiral != yTr_spiral)
print("Training error: %2.4f" % trainingerr_spiral)

predsTe_spiral = svmclassify_spiral(xTe_spiral)
testerr_spiral = np.mean(predsTe_spiral != yTe_spiral)
print("Test error: %2.4f" % testerr_spiral)


# Now, let's visualize the classifier on the spiral dataset!

# In[51]:


visclassifier(svmclassify_spiral, xTr_spiral, yTr_spiral)


# <h3>Interactive Demo</h3>
# 
# Running the code below will create an interactive window where you can click to add new data points to see how a kernel SVM with RBF kernel will respond. There may be a significant delay between clicks.

# In[53]:


Xdata = []
ldata = []
svmC=10;

fig = plt.figure()
details = {
    'ax': fig.add_subplot(111), 
}

plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Click to add positive point and shift+click to add negative points.')

def vis2(fun,xTr,yTr):
    yTr = np.array(yTr).flatten()
    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    classvals = np.unique(yTr)
    res=150
    xrange = np.linspace(0,1,res)
    yrange = np.linspace(0,1,res)
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T
    testpreds = fun(xTe)
    Z = testpreds.reshape(res, res)
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
           )
    plt.show()


def generate_onclick(Xdata, ldata):    
    global details
    def onclick(event):
        if event.key == 'shift': 
            # add positive point
            details['ax'].plot(event.xdata,event.ydata,'or')
            label = 1
        else: # add negative point
            details['ax'].plot(event.xdata,event.ydata,'ob')
            label = -1    
        pos = np.array([event.xdata, event.ydata])
        ldata.append(label)
        Xdata.append(pos)
        
        X=np.array(Xdata)
        Y=np.array(ldata)
        beta_sol, bias_sol, final_loss = minimize(objective=loss, grad=grad, xTr=X, yTr=Y, C=svmC, kerneltype='rbf', kpar=1)
        svmclassify_demo = lambda x: np.sign(computeK('rbf', X, x, 1).transpose().dot(beta_sol) + bias_sol)
        vis2(svmclassify_demo, X, Y)    
    return onclick


cid = fig.canvas.mpl_connect('button_press_event', generate_onclick(Xdata, ldata))
plt.show()


# ### Scikit-learn Implementation
# 
# Scikit-learn provides a variety of kernels to create [kernel SVM classifiers](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). Here's an example on the spiral dataset.

# In[54]:


from sklearn.svm import SVC

clf = SVC(
    C=100,
    kernel='rbf',
    gamma=1, # equivalent to kpar in our implementation
    shrinking=False,
    tol=1e-8, # early stopping threshold, solver stops when successive losses don't change more than tol
    max_iter=10000,
    random_state=0
)
clf.fit(xTr_spiral, yTr_spiral)

predsTr_spiral = clf.predict(xTr_spiral)
trainingerr_spiral = np.mean(predsTr_spiral != yTr_spiral)
print("Training error: %2.4f" % trainingerr_spiral)

predsTe_spiral = clf.predict(xTe_spiral)
testerr_spiral = np.mean(predsTe_spiral != yTe_spiral)
print("Test error: %2.4f" % testerr_spiral)


# In[ ]:




