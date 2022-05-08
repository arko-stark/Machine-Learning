import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



# DATA FRAME

'''
load the iris dataset into a DataFrame, 
which is essentially a NumPy array with column names, 
useful attributes and built-in functions. 
You'll explore the dataset using pandas.
'''
iris = load_iris()
df = pd.DataFrame(data=iris.data,
                  columns=iris.feature_names)

print(df)

# HEAD FUNCTION

'''
The function head(n) allows you to preview 
the first "n" rows of the DataFrame

'''
n = df.head(12)
print(n)


# DESCRIBE FUNCTION

'''
The function .describe() gives us a useful 
quantitative summary of the values in each column
'''
ds = df.describe()
print(ds)

# SHAPE = Similar to numpy
sp = df.shape
print(sp)

# Identify MISSING VALUES
'''
When using real-world datasets, we'll often encounter 
missing values. We can see how many values are missing 
in each column of our data using is.na() 
and summing across the columns:
'''
mc = df.isna().sum(axis = 0) # missing values across columns
print(mc)

mr = df.isna().sum(axis=1) # missing values across rows
print(mr)

# CORRELATION Between Features
'''
Before building a model to predict the target variable, 
it's often useful to look at the distributions of the features 
as well as any correlation structure that might exist between them. 
`pandas` gives us an easy way to do this using .corr()
'''
cr = df.corr()
print(cr)

# DISTRIBUTION OF A FEATURE
'''
We see above that petal length and petal width are strongly 
correlated, while sepal length and sepal width are weakly 
correlated. Let's look at the 
distribution of a few of these features by using .hist()
'''
hist_sl = df['sepal length (cm)'].hist()
print(hist_sl)
hist_pl = df['petal length (cm)'].hist();
print(hist_pl)