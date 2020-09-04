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
from timeit import default_timer as timer

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

# Over the test collection
Accuracy_NNN = []
Recall_NNN = []
precision_NNN = []

Accuracy_LLL = []
Recall_LLL = []
precision_LLL = []

Accuracy_NBB = []
Recall_NBB = []
precision_NBB = []

Accuracy_RDD = []
Recall_RDD = []
precision_RDD = []

# Over the training collection 
Accuracy_NNN_t = []
Recall_NNN_t = []
precision_NNN_t = []

Accuracy_LLL_t = []
Recall_LLL_t = []
precision_LLL_t = []

Accuracy_NBB_t = []
Recall_NBB_t = []
precision_NBB_t = []

Accuracy_RDD_t = []
Recall_RDD_t = []
precision_RDD_t = []
# Runtime over test set
Runtime_NNN = []
Runtime_LLL= []
Runtime_NBB = []
Runtime_RDD = []

# Runtime over training set
Runtime_NNN_t = []
Runtime_LLL_t= []
Runtime_NBB_t = []
Runtime_RDD_t = []

for i in range(20):
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
    
    NN_start_t = timer()
    # Training model
    clf.fit(Xtrain, ytrain_)   
    Runtime_NNN_t.append(timer()-NN_start_t)    
    
    # Predict from training model
    #print(clf.predict(Xtrain))
    
    # Find f1_score for neural network
    #ypred_Logistic = clf.predict(Xtest)
    #f1_score = f1_score(ytest, ypred_Logistic, average='weighted')
    #print('f1 score {0}'.format(f1_score))
    
    print("Neural network - The training score: %.3f" % (clf.score(Xtrain, ytrain_)))
    print('Neural network - The test score is {0}'.format(clf.score(Xtest, ytest_)))
    
    
    # Multiclass Logistic regression with stochastic gradient descent
    clf_SGD = SGDClassifier(loss="log", max_iter=1000, shuffle = True,
                        learning_rate = 'constant', eta0 = 0.01, 
                        average = False, l1_ratio =0,
                        penalty ='none', power_t = 1)
    
    SGD_start_t = timer()
    # Training model
    clf_SGD.fit(Xtrain, ytrain_)
    Runtime_LLL_t.append(timer()-SGD_start_t)
    
    # Predict model
    #ypred_Logistic = clf.predict(Xtest)
    
    print("Multinomial Logistic Regression - The training score: %.3f (%s)" % (clf_SGD.score(Xtrain, ytrain_), 'SGD multinomial'))
    print("Multinomial Logistic Regression - The testing score: %.3f (%s)" % (clf_SGD.score(Xtest, ytest_), 'SGD multinomial'))
    #print("Logistic - Number of mislabeled points out of a total %d points : %d" % (trainset[0].shape[0],(ytest_ != ypred_Logistic).sum()))
    
    # Gaussian Naive bayes 
    clf_NB = GaussianNB()
    NB_start_t = timer()
    clf_NB.fit(Xtrain, ytrain_)    
    Runtime_NBB_t.append(timer()-NB_start_t)
    #print("Number of mislabeled points out of a total %d points : %d" % (Xtrain.shape[0],(ytest_ != y_pred).sum()))
    print("Naive Bayes - training score: %.3f (%s)" % (clf_NB.score(Xtrain, ytrain_), 'Naive Bayes'))
    print("Naive Bayes - testing score: %.3f (%s)" % (clf_NB.score(Xtest, ytest_), 'Naive Bayes'))

    # Random predictor
    clf_random = DummyClassifier(strategy = 'uniform')
    RD_start_t = timer()
    clf_random.fit(Xtrain, ytrain_)
    Runtime_RDD_t.append(timer()-RD_start_t)
    print("Random Predictor - testing score: %.3f (%s)" % (clf_random.score(Xtest, ytest_), 'Random predictor'))
    
    # Prediction over test set
    NN_start = timer()
    ypred_NN=clf.predict(Xtest)
    Runtime_NNN.append(timer()-NN_start)
    
    SGD_start = timer()
    ypred_Logistic=clf_SGD.predict(Xtest)
    Runtime_LLL.append(timer()-SGD_start)
    
    NB_start = timer()
    ypred_NB = clf_NB.predict(Xtest)
    Runtime_NBB.append(timer()-NB_start)
    
    RD_start = timer()
    ypred_RD = clf_random.predict(Xtest)
    Runtime_RDD.append(timer()-RD_start)
    
    # Prediction over training set
    ypred_NN_t=clf.predict(Xtrain)
    ypred_Logistic_t=clf_SGD.predict(Xtrain)
    ypred_NB_t = clf_NB.predict(Xtrain)
    ypred_RD_t = clf_random.predict(Xtrain)
    
    print(60*'-')
    
    # Over the test set
    accuracy_NN = accuracy_score(ytest_, ypred_NN)
    precision_NN=precision_score(ytest_, ypred_NN, average='macro')
    recall_NN = recall_score(ytest_, ypred_NN, average='macro')
    print("Accuracy call for Neural Network: {0}".format(accuracy_NN))
    print("Precision call for Neural Network: {0}".format(precision_NN))
    print("Recall call for Neural Network: {0}".format(recall_NN))
    
    accuracy_Lg = accuracy_score(ytest_, ypred_Logistic)
    precision_Lg=precision_score(ytest_, ypred_Logistic, average='macro')
    recall_Lg=recall_score(ytest_, ypred_Logistic, average='macro')
    print("Accuracy call for Logistic: {0}".format(accuracy_Lg))
    print("Precision call for Logistic: {0}".format(precision_Lg))
    print("Recall call for Logistic regression: {0}".format(recall_Lg))
    
    
    accuracy_NB = accuracy_score(ytest_, ypred_NB)
    precision_NB=precision_score(ytest_, ypred_NB, average='macro')
    recall_NB=recall_score(ytest_, ypred_NB, average='macro')
    print("Accuracy call for Naive Bayes: {0}".format(accuracy_NB))
    print("Precision call for Naive Bayes: {0}".format(precision_NB))
    print("Recall call for Naive Bayes: {0}".format(recall_NB))
    
    accuracy_RD = accuracy_score(ytest_, ypred_RD)
    precision_RD=precision_score(ytest_, ypred_RD, average='macro')
    recall_RD=recall_score(ytest_, ypred_RD, average='macro')
    
    print(60*'-')
    
    # Over the training set
    accuracy_NN_t = accuracy_score(ytrain_, ypred_NN_t)
    precision_NN_t=precision_score(ytrain_, ypred_NN_t, average='macro')
    recall_NN_t = recall_score(ytrain_, ypred_NN_t, average='macro')
    
    accuracy_Lg_t = accuracy_score(ytrain_, ypred_Logistic_t)
    precision_Lg_t=precision_score(ytrain_, ypred_Logistic_t, average='macro')
    recall_Lg_t=recall_score(ytrain_, ypred_Logistic_t, average='macro')
    
    accuracy_NB_t = accuracy_score(ytrain_, ypred_NB_t)
    precision_NB_t=precision_score(ytrain_, ypred_NB_t, average='macro')
    recall_NB_t=recall_score(ytrain_, ypred_NB_t, average='macro')
    
    accuracy_RD_t = accuracy_score(ytrain_, ypred_RD_t)
    precision_RD_t=precision_score(ytrain_, ypred_RD_t, average='macro')
    recall_RD_t=recall_score(ytrain_, ypred_RD_t, average='macro')
    
    # Update the collection over the test set
    Accuracy_NNN.append(accuracy_NN)
    Recall_NNN.append(recall_NN)
    precision_NNN.append(precision_NN)
    
    Accuracy_LLL.append(accuracy_Lg)
    Recall_LLL.append(recall_Lg)
    precision_LLL.append(precision_Lg)
    
    Accuracy_NBB.append(accuracy_NB)
    Recall_NBB.append(recall_NB)
    precision_NBB.append(precision_NB)
    
    Accuracy_RDD.append(accuracy_RD)
    Recall_RDD.append(recall_RD)
    precision_RDD.append(precision_RD)
    
    
    # Update the collection over the training set
    Accuracy_NNN_t.append(accuracy_NN_t)
    Recall_NNN_t.append(recall_NN_t)
    precision_NNN_t.append(precision_NN_t)
    
    Accuracy_LLL_t.append(accuracy_Lg_t)
    Recall_LLL_t.append(recall_Lg_t)
    precision_LLL_t.append(precision_Lg_t)
    
    Accuracy_NBB_t.append(accuracy_NB_t)
    Recall_NBB_t.append(recall_NB_t)
    precision_NBB_t.append(precision_NB_t)
    
    Accuracy_RDD_t.append(accuracy_RD_t)
    Recall_RDD_t.append(recall_RD_t)
    precision_RDD_t.append(precision_RD_t)

print(60*'-')
print('The results below over test set')

print("Mean Accuracy call for Neural Network: {0}".format(np.mean(Accuracy_NNN)))
print("Mean Precision call for Neural Network: {0}".format(np.mean(precision_NNN)))
print("Mean Recall call for Neural Network: {0}".format(np.mean(Recall_NNN)))

print("Mean Accuracy call for Logistic: {0}".format(np.mean(Accuracy_LLL)))
print("Mean Precision call for Logistic: {0}".format(np.mean(Recall_LLL)))
print("Mean Recall call for Logistic regression: {0}".format(np.mean(precision_LLL)))

print("Mean Accuracy call for Naive Bayes: {0}".format(np.mean(Accuracy_NBB)))
print("Mean Precision call for Naive Bayes: {0}".format(np.mean(precision_NBB)))
print("Mean Recall call for Naive Bayes: {0}".format(np.mean(Recall_NBB)))

print("Mean Accuracy call for Random Predictor: {0}".format(np.mean(Accuracy_RDD)))
print("Mean Precision call for Random Predictor: {0}".format(np.mean(precision_RDD)))
print("Mean Recall call for Random Predictor: {0}".format(np.mean(Recall_RDD)))

print(20*'-')
print('The results below over trainning set')
print("T - Mean Accuracy call for Neural Network: {0}".format(np.mean(Accuracy_NNN_t)))
print("T- Mean Precision call for Neural Network: {0}".format(np.mean(precision_NNN_t)))
print("T- Mean Recall call for Neural Network: {0}".format(np.mean(Recall_NNN_t)))

print("T- Mean Accuracy call for Logistic: {0}".format(np.mean(Accuracy_LLL_t)))
print("T- Mean Precision call for Logistic: {0}".format(np.mean(Recall_LLL_t)))
print("T- Mean Recall call for Logistic regression: {0}".format(np.mean(precision_LLL_t)))

print("T- Mean Accuracy call for Naive Bayes: {0}".format(np.mean(Accuracy_NBB_t)))
print("T- Mean Precision call for Naive Bayes: {0}".format(np.mean(precision_NBB_t)))
print("T- Mean Recall call for Naive Bayes: {0}".format(np.mean(Recall_NBB_t)))

print("T- Mean Accuracy call for Random Predictor: {0}".format(np.mean(Accuracy_RDD_t)))
print("T- Mean Precision call for Random Predictor: {0}".format(np.mean(precision_RDD_t)))
print("T- Mean Recall call for Random Predictor: {0}".format(np.mean(Recall_RDD_t)))

print("T- Mean Run time - Training for Neural Network: {0}".format(np.mean(Runtime_NNN_t)))
print("T- Mean Run time - Training for Logistic Regression: {0}".format(np.mean(Runtime_LLL_t)))
print("T- Mean Run time - Training for Naive Bayes: {0}".format(np.mean(Runtime_NBB_t)))
print("T- Mean Run time - Training for Random Predictor: {0}".format(np.mean(Runtime_RDD_t)))


print("Mean Run time - Prediction for Neural Network: {0}".format(np.mean(Runtime_NNN)))
print("Mean Run time - Prediction for Logistic Regression: {0}".format(np.mean(Runtime_LLL)))
print("Mean Run time - Prediction for Naive Bayes: {0}".format(np.mean(Runtime_NBB)))
print("Mean Run time - Prediction for Random Predictor: {0}".format(np.mean(Runtime_RDD)))


def autolabel(rectangles):
    """attach some text vi autolabel on rectangles."""
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height, '%.4f' % height,
                ha='center', va='bottom')
        plt.setp(plt.xticks()[1], rotation=30)

#bar_colors = ['b','g','r','c','m','y']
## Plot figures for runtime over training data set
#plt.figure()
#cls_names = ["Neural Network", "Multinomial Logistic Regression", "Naive Bayes"]
#cls_runtime = []
#cls_runtime.append(np.mean(Runtime_NNN_t))
#cls_runtime.append(np.mean(Runtime_LLL_t))
#cls_runtime.append(np.mean(Runtime_NBB_t))
##cls_runtime.append(2.236502099)
##cls_runtime.append(1.1119242)
##cls_runtime.append(0.008482879)
#ax = plt.subplot()
#rectangles = plt.bar(range(len(cls_names)), cls_runtime, width = 0.5,
#                     color = bar_colors)
#ax.set_xticks(np.linspace(0, len(cls_names)-1, len(cls_names)))
#ax.set_xticklabels(cls_names, fontsize = 8)
#ymax = max(cls_runtime)*1.3
#ax.set_ylim((0,ymax))
#plt.ylabel('Runtime (s)')
#plt.title('Training Times')
#autolabel(rectangles)
#plt.show()
#
#
## Plot figures for runtime over prediction
#plt.figure()
#cls_names = ["Neural Network", "Multinomial Logistic Regression", "Naive Bayes"]
#cls_runtime = []
#cls_runtime.append(np.mean(Runtime_NNN))
#cls_runtime.append(np.mean(Runtime_LLL))
#cls_runtime.append(np.mean(Runtime_NBB))
##cls_runtime.append(0.00450972)
##cls_runtime.append(0.000755058)
##cls_runtime.append(0.003842247)
#ax = plt.subplot()
#rectangles = plt.bar(range(len(cls_names)), cls_runtime, width = 0.5,
#                     color = bar_colors)
#ax.set_xticks(np.linspace(0, len(cls_names)-1, len(cls_names)))
#ax.set_xticklabels(cls_names, fontsize = 8)
#ymax = max(cls_runtime)*1.3
#ax.set_ylim((0,ymax))
#plt.ylabel('Runtime (s)')
#plt.title('Prediction Times')
#autolabel(rectangles)
#plt.show()


print(60*'-')
## Paired-t test
# Between Neural network - Logistic regression
ttest_ac_1 = stats.ttest_rel(Accuracy_NNN,Accuracy_LLL)
ttest_pre_1 = stats.ttest_rel(precision_NNN,precision_LLL)
ttest_re_1 = stats.ttest_rel(Recall_NNN,Recall_LLL)
ttest_run_1 = stats.ttest_rel(Runtime_NNN,Runtime_LLL)
print('Accuracy - Pair-t test between Neural and Logistic {0}'.format(ttest_ac_1))
print('Precision - Pair-t test between Neural and Logistic {0}'.format(ttest_pre_1))
print('Recall - Pair-t test between Neural and Logistic {0}'.format(ttest_re_1))
print('Runtime - Pair-t test between Neural and Logistic {0}'.format(ttest_run_1))
print(60*'-')

# Between Neural network - Naive Bayes
ttest_ac_2 = stats.ttest_rel(Accuracy_NNN,Accuracy_NBB)
ttest_pre_2 = stats.ttest_rel(precision_NNN,precision_NBB)
ttest_re_2 = stats.ttest_rel(Recall_NNN,Recall_NBB)
ttest_run_2 = stats.ttest_rel(Runtime_NNN,Runtime_NBB)
print('Accuracy - Pair-t test between Neural and Naive {0}'.format(ttest_ac_2))
print('Precision - Pair-t test between Neural and Naive {0}'.format(ttest_pre_2))
print('Recall - Pair-t test between Neural and Naive {0}'.format(ttest_re_2))
print('Runtime - Pair-t test between Neural and Naive {0}'.format(ttest_run_2))
print(60*'-')

# Between Logistic regression - Naive Bayes
ttest_ac_3 = stats.ttest_rel(Accuracy_LLL,Accuracy_NBB)
ttest_pre_3 = stats.ttest_rel(precision_LLL,precision_NBB)
ttest_re_3 = stats.ttest_rel(Recall_LLL,Recall_NBB)
ttest_run_3 = stats.ttest_rel(Runtime_LLL,Runtime_NBB)
print('Accuracy - Pair-t test between Logistic and Naive {0}'.format(ttest_ac_3))
print('Precision - Pair-t test between Logistic and Naive {0}'.format(ttest_pre_3))
print('Recall - Pair-t test between Logistic and Naive {0}'.format(ttest_re_3))
print('Runtime - Pair-t test between Logistic and Naive {0}'.format(ttest_run_3))

def autolabel(rectangles):
    """attach some text vi autolabel on rectangles."""
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height, '%.4f' % height,
                ha='center', va='bottom')
        plt.setp(plt.xticks()[1], rotation=30)