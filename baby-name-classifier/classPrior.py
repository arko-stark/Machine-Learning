'''
In the project, you will need to write a function that determines the
class priors for baby names. For example, if you have a simple array of
binary labels that represents a training dataset of baby names,
where “0” indicates a boy name and “1” indicates a girl name, your sample might look like this:


[0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0]


Given this example array, you could easily count the number of occurrences and
determine that the class prior for boy names is 7/12 and for girl names is 5/12.
But what if you have a dataset that is much larger? How might you calculate the class prior efficiently?
'''




import numpy as np
def class_prior(labels) :
    # count_total = np.size(labels) # not required since we are handling a binary label and tacking this with np.mean
    prior_girl = np.float64(np.mean(labels))
    prior_boy = 1 - prior_girl
    return prior_girl, prior_boy






label1 = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0]
print(class_prior(label1))