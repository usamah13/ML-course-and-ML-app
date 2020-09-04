from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

## k-fold cross-validation
# K - number of folds
# X - data to partition
# Y - targets to partition
# classalgs - a dictionary mapping algorithm names to algorithm instances
#
# example:
# classalgs = {
#   'nn_0.01': algs.NeuralNet({ 'regwgt': 0.01 }),
#   'nn_0.1':  algs.NeuralNet({ 'regwgt': 0.1  }),
# }
 
#def cross_validate(K, X, Y, classalgs):
def cross_validate(K, X, classalgs):
#    for k in range(K):
#        for learnername in classalgs:
#            print('make this work')
    
    nsamples = X[0].shape[0]    
    interval = int(nsamples/K)
    print('interval is {0}'.format(interval))    
    bestAccuracy = 0
    bestLearner = ""
    errors = {}
    for learnername, learner in classalgs.items():  
        
        errors[learnername] = np.zeros(K)
        
        accuracy = 0                
        for k in range(0, K):
            trainset = [1,1]
            testset = [1,1]
            if k == 0:
                trainset[0] = X[0][interval:,:]
                trainset[1] = X[1][interval:]
                testset[0] = X[0][:interval,:]
                testset[1] = X[1][:interval] 
            elif k == K-1:
                trainset[0] = X[0][:(K-1)*interval,:]
                trainset[1] = X[1][:(K-1)*interval]
                testset[0] = X[0][(K-1)*interval:,:]
                testset[1] = X[1][(K-1)*interval:]   
            else:
                trainset1 = X[0][: k*interval,:]
                trainset2 = X[0][(k+1)*interval : ,:]
                trainset[0] = np.concatenate((trainset1, trainset2), axis=0)
                trainset1 = X[1][: k*interval]
                trainset2 = X[1][(k+1)*interval :]
                trainset[1] = np.concatenate((trainset1, trainset2), axis=0)
                testset[0] = X[0][k*interval:(k+1)*interval,:]
                testset[1] = X[1][k*interval:(k+1)*interval]     
            # Learn model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(testset[0])            
            accuracy_k = getaccuracy(testset[1], predictions)*interval/100
            accuracy += accuracy_k
            error_k = geterror(testset[1], predictions)
            errors[learnername][k] = error_k
#            print('accuracy run {0} = {1}'.format(k, accuracy_k))
        
#        print("k-fold cross validation = " + learnername)        
#        print("Accuracy= " + str(accuracy))
        if bestAccuracy < accuracy:
                bestAccuracy = accuracy
                bestLearner = learnername
        
        # Additionally report the standard error over multiple runs over the test set
        std_error = np.std(errors[learnername][:])/math.sqrt(K)
        print('k-fold Standard error for '+ learnername +' is: ' + str(std_error))
                        
        print("k-fold Average error for " + learnername +' is: ' + str(np.sum(errors[learnername])/K))
    
#    print("Best accuracy= " + str(bestAccuracy))
    print("Best_algorithm from k-fold= " + bestLearner)
    best_algorithm = classalgs[bestLearner]
    return best_algorithm


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 1

    classalgs = {
                 #'Random': algs.Classifier(),
                 #'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 #'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 #'Linear Regression': algs.LinearRegressionClass(),
                 #'Logistic Regression': algs.LogitReg(),
                 #'Neural Network': algs.NeuralNet({'epochs': 100})
                 'Kernel': algs.KernelLogitReg({'kernel': "linear"})
                }
#    classalgs = {
##       'nn_0.01': algs.LinearRegressionClass({ 'regwgt': 0.01 }),
#       #'nn_0.1':  algs.LinearRegressionClass({ 'regwgt': 0.1  }),
#       'lg_0.1': algs.LogitReg({ 'stepsize': 0.1 }),
#       'lg_0.01': algs.LogitReg({ 'stepsize': 0.01 }),
#       'lg_0.001': algs.LogitReg({ 'stepsize': 0.001 }),
#     }
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 4},
#        {'regwgt': 0.01, 'nh': 8},
#        {'regwgt': 0.05, 'nh': 16},
#        {'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        #trainset, testset = dtl.load_census(trainsize,testsize)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error
        
        
        # Running k-fold internal cross validation
        #cross_validate(10, trainset, classalgs)

    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
