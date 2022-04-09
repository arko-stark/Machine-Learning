'''
In the final project of this course, you will need to write a function that
assumes conditional independence and determines class-conditional probabilities.
Imagine you have the following vector of probabilities of 5 features given a class label c:
P(x|y=c) = [0.212, 0.234, 0.155, .021, .04]

Since you will assume independence, you can simply multiply these probabilities to get the class
conditional probability of observing this data point (i.e. this combination of feature values).

'''
import numpy as np
def sum_log(cond_prob_features):
    return np.sum(np.log(cond_prob_features))

# testing the function
cond_prob = np.array([0.212, 0.234, 0.155, .021, .04])
print(sum_log(cond_prob))
