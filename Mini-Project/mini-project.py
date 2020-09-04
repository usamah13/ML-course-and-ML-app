# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:43:56 2018

@author: phuongnh
"""
from math import gcd
import numpy as np
import dataloader as dtl
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Library for statistical significant test
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load data from sklearn - classification - digits
#digits = load_digits()
#bigdata = [digits.data, digits.target]

# Load data from csv file
#trainset = dtl.load_cardiac_trainset(190)
#testset = dtl.load_cardiac_testset(190)

bigdata = dtl.load_cardiac_trainset(380)

# shuffle
#idx = np.arange(bigdata[0].shape[0])
idx = list(range(380))
np.random.shuffle(idx)
X = bigdata[0][idx]
y = bigdata[1][idx]

#trainset = dtl.load_displacement_trainset(190)
#testset = dtl.load_displacement_testset(190)

Xtrain = preprocessing.scale(X[:190,:])
#Xtrain = X[:190,:]
ytrain = y[:190]
#ytrain = y[:190,:]
#print(Xtrain)

#Xtrain = Xtrain[:, :1]
#print(ytrain)

Xtest = preprocessing.scale(X[190:,:])
#Xtest = X[190:,:]
ytest = y[190:]



# Convert [1,0,0] to 0. [0,1,0] to 1. [0,0,1] to 2
ytrain_ = np.zeros(ytrain.shape[0])
for i in range(ytrain.shape[0]):
    index = np.argmax(ytrain[i])
    if index == 0:
        ytrain_[i] = int(0)
    elif index == 1:
        ytrain_[i] = int(1)
    else:
        ytrain_[i] = int(2)

ytest_ = np.zeros(ytest.shape[0])
for i in range(ytest.shape[0]):
    index = np.argmax(ytest[i])
    if index == 0:
        ytest_[i] = int(0)
    elif index == 1:
        ytest_[i] = int(1)
    else:
        ytest_[i] = int(2)

# Find number of each class
#numclass1 = (ytest_ == 0).sum()
#print(numclass1)


# Neural network with one hidden layer - 16 neurons in hidden layer
clf = MLPClassifier(solver='sgd', alpha=0, activation='logistic',
                    batch_size=1,max_iter=1000,
                    learning_rate='constant',learning_rate_init=0.1,
                    shuffle=True,
                    hidden_layer_sizes=(12,))

# Training model
clf.fit(Xtrain, ytrain_)   

# Predict from training model
#print(clf.predict(Xtrain))

# Find f1_score for neural network
#ypred_Logistic = clf.predict(Xtest)
#f1_score = f1_score(ytest, ypred_Logistic, average='weighted')
#print('f1 score {0}'.format(f1_score))

print("Neural network - The training score: %.3f" % (clf.score(Xtrain, ytrain_)))
print('Neural network - The test score is {0}'.format(clf.score(Xtest, ytest_)))


#prediction = np.zeros((Xtrain.shape[0],3))
#   
##print (y_pred)
##prediction = clf.predict([[2., 2.], [-1., -2.]])
##print(y_pred.shape)
#for i in range(y_pred.shape[0]):
#    index = np.argmax(y_pred[i])
#    if index == 0:
#        prediction[i] = np.array([1,0,0])
#    elif index == 1:
#        prediction[i] = np.array([0,1,0])
#    else:
#        prediction[i] = np.array([0,0,1])    
#
#print("Neural network - Number of mislabeled points out of a total %d points : %d" % (trainset[0].shape[0],(ytrain != prediction).sum()))

# Multiclass Logistic regression with stochastic gradient descent
clf_SGD = SGDClassifier(loss="log", max_iter=1000, shuffle = True,
                    learning_rate = 'constant', eta0 = 0.1, 
                    average = False, l1_ratio =0,
                    penalty ='none', power_t = 1)

# Training model
clf_SGD.fit(Xtrain, ytrain_)

# Predict model
#ypred_Logistic = clf.predict(Xtest)

print("Multinomial Logistic Regression - The training score: %.3f (%s)" % (clf_SGD.score(Xtrain, ytrain_), 'SGD multinomial'))
print("Multinomial Logistic Regression - The testing score: %.3f (%s)" % (clf_SGD.score(Xtest, ytest_), 'SGD multinomial'))
#print("Logistic - Number of mislabeled points out of a total %d points : %d" % (trainset[0].shape[0],(ytest_ != ypred_Logistic).sum()))

# Find index of incorrect prediction
#indexxx = np.where(ytest_ != ypred_Logistic)
#print(indexxx)
#print(Xtest[indexxx])

# Gaussian Naive bayes 
clf_NB = GaussianNB()
clf_NB.fit(Xtrain, ytrain_)
ypred_NB = clf_NB.predict(Xtrain)
#print("Number of mislabeled points out of a total %d points : %d" % (Xtrain.shape[0],(ytest_ != y_pred).sum()))
print("Naive Bayes - training score: %.3f (%s)" % (clf_NB.score(Xtrain, ytrain_), 'Naive Bayes'))
print("Naive Bayes - testing score: %.3f (%s)" % (clf_NB.score(Xtest, ytest_), 'Naive Bayes'))


# Random predictor
clf_random = DummyClassifier(strategy = 'uniform')
clf_random.fit(Xtrain, ytrain_)
ypred_random = clf_random.predict(Xtest)
print("Random Predictor - training score: %.3f (%s)" % (clf_random.score(Xtrain, ytrain_), 'Random predictor'))


# Paired t-test
#df = pd.read_csv("datasets/Data-sets-master/blood_pressure.csv")

# Assumption check: Outliers
#df[['bp_before','bp_after']].plot(kind='box')
#print(df[['bp_before','bp_after']])

# Assumption check: normal distribution
#df['bp_difference'] = df['bp_before'] - df['bp_after']
#df['bp_difference'].plot(kind='hist', title= 'Blood Pressure Difference Histogram')
#


# Ask TA about precision and recall

# K-fold cross validation for neural network
# Neural network with one hidden layer - 16 neurons in hidden layer
#clf = MLPClassifier(solver='sgd', alpha=0, activation='logistic',
#                    batch_size=1,max_iter=1000,
#                    learning_rate='constant',learning_rate_init=0.01,
#                    shuffle=True,
#                    hidden_layer_sizes=(16,), random_state=1)

#clf_k = LogisticRegression(solver='sag', max_iter=1,
#                             multi_class='multinomial')







# Stratified k-fold cross validation
accuracy_c = []
clf = MLPClassifier(solver='sgd', alpha=0, activation='logistic',
                    batch_size=1,max_iter=1000,
                    learning_rate='constant',learning_rate_init=0.001,
                    shuffle=True,
                    hidden_layer_sizes=(12,))

skf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in skf.split(Xtrain, ytrain_):    
    X_train, X_test = Xtrain[train_index], Xtrain[test_index]
    y_train, y_test = ytrain_[train_index], ytrain_[test_index]
    # Training model
    clf_k.fit(X_train, y_train)    
    print('Neural network - The test score is {0}'.format(clf_k.score(X_test, y_test)))
    accuracy_c.append(clf_k.score(X_test, y_test))

print('Mean accuracy from k-fold = {0}'.format(np.mean(accuracy_c)))