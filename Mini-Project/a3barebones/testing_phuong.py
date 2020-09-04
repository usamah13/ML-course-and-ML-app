# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:43:56 2018

@author: phuongnh
"""
from math import gcd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier


weights = np.arange(9)

weights = np.where(weights > 3, 100, 20)


def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

#distance = hamming_distance(123,321)
#print(distance)


X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='sgd', alpha=0, activation='logistic',
                    batch_size=1,max_iter=100,
                    learning_rate='constant',learning_rate_init=0.01,
                    shuffle=True,
                    hidden_layer_sizes=(16,), random_state=1)

clf.fit(X, y)      
print(clf)




