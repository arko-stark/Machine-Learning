# coding=utf-8
'''
n the project, you will need to extract and hash features of names.
Let’s say you have the following list of names and you want to determine whether each name in the list
ends with a vowel (a, e, i, o, u, y) or a consonant.
 names = ['Natalia', 'Anastasia', 'Emilia', 'Marie', 'Jonas', 'Jordan', 'Brett']
Given this list, you should return a “1” for “Natalia” since that name ends in a vowel.
Since “Jonas” ends in a consonant, you would return a "0". For the exercise, you’ll practice writing code
 that can extract a binary feature representing whether each name ends in a vowel or a consonant.

'''
import numpy as np
def ending_in_vowel(names):
    name_vowelmatch = [names for names in names if names[-1] in 'aeiouy']
    return np.size(name_vowelmatch)


# test code
names = ['Natalia', 'Anastasia', 'Emilia', 'Marie', 'Jonas', 'Jordan', 'Brett']
print(ending_in_vowel(names))