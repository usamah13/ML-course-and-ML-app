from __future__ import division  # floating point division
import math
import numpy as np

####### Main load functions

def load_cardiac_trainset(trainsize=500):
    """ A physics classification dataset """
    filename = 'datasets/Voltagetarinandtestcomide380.csv'
    dataset = loadcsv(filename)
    
    trainset = splitdataset(dataset,trainsize)    
    return trainset

def load_cardiac_testset(testsize=500):
    """ A physics classification dataset """
    filename = 'datasets/testing data.csv'
    dataset = loadcsv(filename)
    
    testset = splitdataset(dataset,testsize)    
    return testset

def load_displacement_trainset(trainsize=500):
    """ A physics classification dataset """
    filename = 'datasets/traindispfinal.csv'
    dataset = loadcsv(filename)
    
    trainset = splitdataset(dataset,trainsize)    
    return trainset

def load_displacement_testset(testsize=500):
    """ A physics classification dataset """
    filename = 'datasets/testdispfinal.csv'
    dataset = loadcsv(filename)
    
    testset = splitdataset(dataset,testsize)    
    return testset


####### Helper functions

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    featureend = dataset.shape[1]-3    
    outputlocation = featureend            
    
    Xtrain = dataset[:trainsize,:featureend]
    ytrain = dataset[:trainsize,outputlocation:]
    
    
    
    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility    
    
                        
                              
    return (Xtrain,ytrain)


def create_susy_dataset(filenamein,filenameout,maxsamples=100000):
    dataset = np.genfromtxt(filenamein, delimiter=',')
    y = dataset[0:maxsamples,0]
    X = dataset[0:maxsamples,1:9]
    data = np.column_stack((X,y))
    
    np.savetxt(filenameout, data, delimiter=",")
