#!/usr/bin/env python
# coding: utf-8

# <h2>About this Project</h2>
# 
# In this project, you will implement bagging for regression trees.
# 
# <h3>Evaluation</h3>
# 
# <p><strong>This project must be successfully completed and submitted in order to receive credit for this course. Your score on this project will be included in your final grade calculation.</strong><p>
#     
# <p>You are expected to write code where you see <em># YOUR CODE HERE</em> within the cells of this notebook. Not all cells will be graded; code input cells followed by cells marked with <em>#Autograder test cell</em> will be graded. Upon submitting your work, the code you write at these designated positions will be assessed using an "autograder" that will run all test cells to assess your code. You will receive feedback from the autograder that will identify any errors in your code. Use this feedback to improve your code if you need to resubmit. Be sure not to change the names of any provided functions, classes, or variables within the existing code cells, as this will interfere with the autograder. Also, remember to execute all code cells sequentially, not just those you’ve edited, to ensure your code runs properly.</p>
#     
# <p>You can resubmit your work as many times as necessary before the submission deadline. If you experience difficulty or have questions about this exercise, use the Q&A discussion board to engage with your peers or seek assistance from the instructor.<p>
# 
# <p>Before starting your work, please review <a href="https://s3.amazonaws.com/ecornell/global/eCornellPlagiarismPolicy.pdf">eCornell's policy regarding plagiarism</a> (the presentation of someone else's work as your own without source credit).</p>
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
# </ol>
# <p>You can also download a copy of this notebook in multiple formats using the <strong>Download as</strong> option in the <strong>File</strong> menu above.</p>

# ### Getting Started
# 
# Before you get started, let's import a few packages that you will need.

# In[1]:


import numpy as np
from pylab import *
from numpy.matlib import repmat
import sys
import matplotlib 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *
get_ipython().run_line_magic('matplotlib', 'notebook')

print('You\'re running python %s' % sys.version.split(' ')[0])


# In this project, we will work with an artificial 2D spiral dataset of size 150 (for visualization), and a high dimensional dataset [ION](https://archive.ics.uci.edu/ml/datasets/Ionosphere) (for a binary test classification problem).

# In[3]:


xTrSpiral, yTrSpiral, xTeSpiral, yTeSpiral = spiraldata(150)
xTrIon, yTrIon, xTeIon, yTeIon = iondata()


# You will use the regression tree from a previous project. As a reminder, the following code shows you how to instantiate a decision tree:

# In[4]:


# Create a regression tree with no restriction on its depth
# and weights for each training example to be 1
# if you want to create a tree of max_depth k
# then call RegressionTree(depth=k)
tree = RegressionTree(depth=np.inf)

# To fit/train the regression tree
tree.fit(xTrSpiral, yTrSpiral)

# To use the trained regression tree to predict a score for the example
score = tree.predict(xTrSpiral)

# To use the trained regression tree to make a +1/-1 prediction
pred = np.sign(tree.predict(xTrSpiral))
        
tr_err   = np.mean((np.sign(tree.predict(xTrSpiral)) - yTrSpiral)**2)
te_err   = np.mean((np.sign(tree.predict(xTeSpiral)) - yTeSpiral)**2)

print("Training error: %.4f" % np.mean(np.sign(tree.predict(xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(tree.predict(xTeSpiral)) != yTeSpiral))


# <p>The following code defines a function <code>visclassifier()</code>, which plots the decision boundary of a classifier in 2 dimensions. Execute the following code to see what the decision boundary of your tree looks like on the spiral data set. </p>

# In[5]:


def visclassifier(fun,xTr,yTr):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """

    yTr = np.array(yTr).flatten()
    
    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    # get the unique values from labels array
    classvals = np.unique(yTr)

    plt.figure()

    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # test all of these points on the grid
    testpreds = fun(xTe)
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    
    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # creates x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
                   )

    plt.axis('tight')
    # shows figure and blocks
    plt.show()
    

tree=RegressionTree(depth=np.inf)
tree.fit(xTrSpiral,yTrSpiral) # compute tree on training data 
visclassifier(lambda X: tree.predict(X),xTrSpiral,yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(tree.predict(xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(tree.predict(xTeSpiral)) != yTeSpiral))


# ## Bagging in Action
# 
# ### Part One: Implement Bagging [Graded]
# 
# CART trees are known to be high variance classifiers if trained to full depth. Equivalently, CART trees of full depth easily overfit to the training set, which prevent the trees from performing well on new unseen data. An effective way to prevent overfitting is to use **Bagging** (short for **b**ootstrap **ag**gregating). Implement the function **`forest`**, which builds a forest of `m` regression trees of `depth=maxdepth`. Each tree should be built using training data drawn by randomly sampling `n` examples from the training data with replacement, where `n` is the number of points in `xTr`.
# 
# We are going to keep it simple and **not** randomly subsample a small set of features to split on. Therefore, all trees will be constructed with a simple call to `RegressionTree(depth=maxdepth)`. The function should output the list of trees.
# 
# _Hint: You may find [`np.random.choice(a, b, replace=True)`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) useful. It samples `b` numbers from `[0, ..., a-1]` with replacement._

# In[16]:


def forest(xTr, yTr, m, maxdepth=np.inf):
    """
    Creates a forest of m trees, each of depth=maxdepth.
    
    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of each tree
        
    Output:
        trees: list of decision trees of length m
    """
    
    n, d = xTr.shape
    trees = []
    
    # YOUR CODE HERE
    #start the loop for m runs
    for i in range(m):
        # random indices on n training data points
        rand_ind = np.random.choice(n,n,replace = True)
        t1 = RegressionTree(depth = maxdepth)
        # get training data and labels in the random order
        xTree = xTr[rand_ind,:]
        yTree = yTr[rand_ind]
        t1.fit(xTree,yTree)
        trees.append(t1)
    return trees


# In[17]:


# # testing cell, please ignore
# i = np.random.choice(2,2,replace = True)
# # print(i)
# m = 20
# x = np.arange(100).reshape((100, 1))
# y = np.arange(100)
# n,d = x.shape
# print(n)
# i = np.random.choice(n,n,replace = True)
# # print(i)
# xt = x[i,:]
# yt = y[i]
# print(y)


# In[18]:


def forest_test1():
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m)
    return len(trees) == m # make sure there are m trees in the forest

def forest_test2():
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    max_depth = 4
    trees = forest(x, y, m, max_depth)
    depths_forest = np.array([tree.depth for tree in trees]) # Get the depth of all trees in the forest
    return np.all(depths_forest == max_depth) # make sure that the max depth of all the trees is correct


def forest_test3():
    s = set()

    def DFScollect(tree):
        # Do Depth first search to all prediction in the tree
        if tree.left is None and tree.right is None:
            s.add(tree.prediction)
        else:
            DFScollect(tree.right)
            DFScollect(tree.left)

    m = 200
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m);

    lens = np.zeros(m)

    for i in range(m):
        s.clear()
        DFScollect(trees[i].root)
        lens[i] = len(s)

    # Check that about 63% of data is represented in each random sample
    return abs(np.mean(lens) - 100 * (1 - 1 / np.exp(1))) < 2

runtest(forest_test1, 'forest_test1')
runtest(forest_test2, 'forest_test2')
runtest(forest_test3, 'forest_test3')


# In[19]:


# Autograder test cell - worth 1 point
# runs forest_test1


# In[20]:


# Autograder test cell - worth 1 point
# runs forest_test2


# In[21]:


# Autograder test cell - worth 1 point
# runs forest_test3


# ### Part Two: Implement `evalforest` [Graded]
# 
# Now implement the function **`evalforest`**, which should take as input a set of $m$ trees and a set of $n$ test inputs and return the **average** prediction of all the trees.
# 
# Note that for bagging, we take the average over all trees weighted equally. In a later project, you will implement a different version of this function that assigns different weights to predictions of different trees.

# In[24]:


def evalforest(trees, X):
    """
    Evaluates X using trees.
    
    Input:
        trees:  list of length m of RegressionTree decision trees
        X:      n x d matrix of data points
        
    Output:
        pred:   n-dimensional vector of predictions
    """
    m = len(trees)
    n,d = X.shape
    
    pred = np.zeros(n)
    
    # YOUR CODE HERE
    #loop through number of trees
    for i in range(m) :
        preds = trees[i].predict(X) # get your predictions from the tree given X
        pred = pred+preds # collect your predictions
    pred = pred/m # calculate the average to get your bagged prediction
    return pred


# In[25]:


# m = 200
# x = np.arange(100).reshape((100, 1))
# y = np.arange(100)
# preds = 0
# trees = forest(x, y, m)
# preds = trees[2].predict(x)
# print(preds)


# In[26]:


def evalforest_test1():
    m = 200
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m)
    
    preds = evalforest(trees, x)
    return preds.shape == y.shape

def evalforest_test2():
    m = 200
    x = np.ones(10).reshape((10, 1))
    y = np.ones(10)
    max_depth = 0
    
    # Create a forest with trees depth 0
    # Since the data are all ones, each tree will return 1 as prediction
    trees = forest(x, y, m, 0) 
    pred = evalforest(trees, np.ones((1, 1)))[0]
    return np.isclose(pred,1) # the prediction should be equal to the sum of weights
    
def bagging_test1():
    m = 50
    xTr = np.random.rand(500,3) - 0.5
    yTr = np.sign(xTr[:,0] * xTr[:,1] * xTr[:,2]) # XOR Classification
    xTe = np.random.rand(50,3) - 0.5
    yTe = np.sign(xTe[:,0] * xTe[:,1] * xTe[:,2])

    tree = RegressionTree(depth=5)
    tree.fit(xTr, yTr)
    oneacc = np.sum(np.sign(tree.predict(xTe)) == yTe)

    trees = forest(xTr, yTr, m, maxdepth=5)
    multiacc = np.sum(np.sign(evalforest(trees, xTe)) == yTe)

    # Check that bagging yields improvement - or doesn't get too much worse
    return multiacc * 1.1 >= oneacc

def bagging_test2():
    m = 50
    xTr = (np.random.rand(500,3) - 0.5) * 4
    yTr = xTr[:,0] * xTr[:,1] * xTr[:,2] # XOR Regression
    xTe = (np.random.rand(50,3) - 0.5) * 4
    yTe = xTe[:,0] * xTe[:,1] * xTe[:,2]
    
    np.random.seed(1)
    tree = RegressionTree(depth=3)
    tree.fit(xTr, yTr)
    oneerr = np.sum(np.sqrt((tree.predict(xTe) - yTe) ** 2))

    trees = forest(xTr, yTr, m, maxdepth=3)
    multierr = np.sum(np.sqrt((evalforest(trees, xTe) - yTe) ** 2))

    # Check that bagging yields improvement - or doesn't get too much worse
    return multierr <= oneerr * 1.5

runtest(evalforest_test1, 'evalforest_test1')
runtest(evalforest_test2, 'evalforest_test2')
runtest(bagging_test1, 'bagging_test1')
runtest(bagging_test2, 'bagging_test2')


# In[27]:


# Autograder test cell - worth 1 point
# runs evalforest-test1


# In[28]:


# Autograder test cell - worth 1 point
# runs evalforest-test2


# In[29]:


# Autograder test cell - worth 1 point
# runs bagging-test1


# In[30]:


# Autograder test cell - worth 1 point
# runs bagging-test2


# ### Visualize the Decision Boundary
# 
# The following script visualizes the decision boundary of an ensemble of decision tree. You might observe that the decision boundary is less rigid with an ensemble of 50 trees than with just 1 CART tree. This is to be expected as a forest is just an ensemble of CART trees and averages the predictions. Consequently, the test error should also be less with the the ensemble than with just 1 CART tree.

# In[31]:


trees=forest(xTrSpiral,yTrSpiral, 50) # compute tree on training data 
visclassifier(lambda X:evalforest(trees,X),xTrSpiral,yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(evalforest(trees,xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(evalforest(trees,xTeSpiral)) != yTeSpiral))


# <h3>Evaluate Test and Training Error</h3>
# 
# <p>The following script evaluates the test and training error of an ensemble of decision trees as we vary the number of trees.</p>

# In[32]:


M=20 # max number of trees
err_trB=[]
err_teB=[]
alltrees=forest(xTrIon,yTrIon,M)
for i in range(M):
    trees=alltrees[:i+1]
    trErr = np.mean(np.sign(evalforest(trees,xTrIon)) != yTrIon)
    teErr = np.mean(np.sign(evalforest(trees,xTeIon)) != yTeIon)
    err_trB.append(trErr)
    err_teB.append(teErr)
    print("[%d]training err = %.4f\ttesting err = %.4f" % (i,trErr, teErr))

plt.figure()
line_tr, = plt.plot(range(M), err_trB, '-*', label="Training error")
line_te, = plt.plot(range(M), err_teB, '-*', label="Testing error")
plt.title("Ensemble of Decision Trees")
plt.legend(handles=[line_tr, line_te])
plt.xlabel("# of trees")
plt.ylabel("error")
plt.show()


# ### 1D Interactive Demo
# 
# The next interactive demo shows a 1-dimensional curve fitted with ensembles of decision trees. We sample 100 training data points from the curve with additive noise. Note how the predicted curve becomes increasingly smooth as you add trees &mdash; adding trees should thus decrease training and testing error. The testing error may be a little higher than the training error, but there is no large gap as ensembles are quite good at avoiding overfitting. 

# In[33]:


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain,yTrain,Q,trees
    
    if event.key == 'shift': Q+=10
    else: Q+=1
    Q=min(Q,M)


    plt.cla()    
    plt.xlim((0,1))
    plt.ylim((0,1))
    pTest=evalforest(trees[:Q],xTest);
    pTrain=evalforest(trees[:Q],xTrain);


    errTrain=np.sqrt(np.mean((pTrain-yTrain)**2))
    errTest=np.sqrt(np.mean((pTest-yTest)**2))

    plt.plot(xTrain[:,0],yTrain,'bx')
    plt.plot(xTest[:,0],pTest,'r-')
    plt.plot(xTest[:,0],yTest,'k-')

    plt.legend(['Training data','Prediction'])
    plt.title('(%i Trees)  Error Tr: %2.4f, Te:%2.4f' % (Q,errTrain,errTest))
    plt.show()
    
        
n=100; # number of training points
NOISE=0.05 # magnitude of noise
xTrain=np.array([np.linspace(0,1,n),np.zeros(n)]).T
yTrain=2*np.sin(xTrain[:,0]*3)*(xTrain[:,0]**2)
yTrain+=np.random.randn(yTrain.size)*NOISE;
ntest=300; # density of test points
xTest=np.array([linspace(0,1,ntest),np.zeros(ntest)]).T
yTest=2*np.sin(xTest[:,0]*3)*(xTest[:,0]**2)



    
# Hyper-parameters (feel free to play with them)
M=100 # number of trees
depth=np.inf
trees=forest(xTrain, yTrain, M,maxdepth=depth)
Q=0;

get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest) 
print('Click to add a tree (shift-click and add 10 trees).');
plt.title('Click to start boosting on this 1D data (Shift-click to add 10 trees).')
plt.plot(xTrain[:,0],yTrain,'*')
plt.plot(xTest[:,0],yTest,'k-')
plt.xlim(0,1)
plt.ylim(0,1)


# <h3>2D Interactive Demo</h3>
# 
# The following demo visualizes the bagged classifier. Add your own points directly on the graph with click and shift+click to see the prediction boundaries. There will be a delay between clicks as the new classifier is trained.

# In[34]:


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain,yTrain,w,b,M
    # create position vector for new point
    pos=np.array([[event.xdata,event.ydata]]) 
    if event.key == 'shift': # add positive point
        color='or'
        label=1
    else: # add negative point
        color='ob'
        label=-1    
    xTrain = np.concatenate((xTrain,pos), axis = 0)
    yTrain = np.append(yTrain, label)
    marker_symbols = ['o', 'x']
    classvals = np.unique(yTrain)
        
    w = np.array(w).flatten()
    
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    
    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(0, 1,res)
    yrange = np.linspace(0, 1,res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # get forest
    trees=forest(xTrain,yTrain,M)
    fun = lambda X:evalforest(trees,X)
    # test all of these points on the grid
    testpreds = fun(xTe)
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    
    plt.cla()    
    plt.xlim((0,1))
    plt.ylim((0,1))
    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
    
    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c,0],
            xTrain[yTrain == c,1],
            marker=marker_symbols[idx],
            color='k'
            )
    plt.show()
    
        
xTrain= np.array([[5,6]])
b=yTrIon
yTrain = np.array([1])
w=xTrIon
M=20

get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
plt.xlim(0,1)
plt.ylim(0,1)
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest)
print('Note: You may notice a delay when adding points to the visualization.')
plt.title('Use shift-click to add negative points.')


# In[ ]:




