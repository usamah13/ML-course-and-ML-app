from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        #print('Parameter is: {0}'.format(self.params))
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)        
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0
        self.classCategories = []

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """
        if(self.params["usecolumnones"] == False):
            Xtrain = np.delete(Xtrain,Xtrain.shape[1]-1,1)

        ### YOUR CODE HERE
        self.classCategories = np.unique(ytrain)
        self.numclasses = self.classCategories.shape[0]
        self.numfeatures = Xtrain.shape[1]
        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE
        for c in range(0, self.numclasses):
            
            index_of_class_c = np.where(ytrain == self.classCategories[c])            
            Xdata = Xtrain[index_of_class_c]
            numpoint_of_class_c = Xdata.shape[0]
            
            # Calculate mean
            sumColums = np.sum(Xdata, axis=0)            
            self.means[c] = sumColums / numpoint_of_class_c
            
            # Calculate the variance (std^2)
            sumColums2 = [sum((Xdata[:,i]- self.means[c][i])**2) for i in range(self.numfeatures)] 
            
            self.stds[c] = np.asarray(sumColums2) / numpoint_of_class_c            
#            print(sumColums)           
#            print("--------------------")
#            print(sumColums2)           
#            print("--------------------")
                  
        ### END YOUR CODE
        
#        print("mean is:")
#        print(self.means)

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        epsilon = 0
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        if(self.params["usecolumnones"] == False):
            Xtest = np.delete(Xtest,Xtest.shape[1]-1,1)

        ### YOUR CODE HERE
        p_x_given_y = np.zeros((Xtest.shape[1], self.numclasses))        
        
        ## Error for division by 0 - That's why adding epsilon
        for i in range(Xtest.shape[0]):
            for j in range(Xtest.shape[1]):
                for c in range(0, self.numclasses):
                    p_x_given_y[j,c] = ((2*math.pi*self.stds[c][j] + epsilon)**-0.5)*math.exp(-((Xtest[i,j]-self.means[c][j])**2)/(2*self.stds[c][j] + epsilon))
            
            max_p_x_given_y = np.ones(self.numclasses)
            for c in range(0, self.numclasses):
                for jj in range(Xtest.shape[1]):
                    max_p_x_given_y[c] = max_p_x_given_y[c]*p_x_given_y[jj,c]
            
            ind = np.unravel_index(np.argmax(max_p_x_given_y, axis=None), max_p_x_given_y.shape)
            ytest[i] = self.classCategories[ind] 
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'stepsize': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE
        
        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE

        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)
        
        ### YOUR CODE HERE
        self.weights = np.random.rand(Xtrain.shape[1],)
        numEpochs = 1000
        stepsize = 0.01
        
#        error = math.inf
#        tolerance = 10**-4
        for i in range(numEpochs):
            
            # Shuffle Xtrain and Ytrain
            Xtrain, ytrain = shuffleData(Xtrain, ytrain)
            
            for j in range(Xtrain.shape[0]):
                
                #Calculate g
                g1 = utils.sigmoid(np.dot(Xtrain[j,:].T,self.weights)) - ytrain[j]                
                g2= Xtrain[j,:]
                g = np.dot(g1,g2)
                self.weights = self.weights - stepsize*g
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        for i in range(Xtest.shape[0]):
            p_1_given_x = 1/(1+math.exp(-np.dot(Xtest[i,:].T,self.weights)))            
            ytest[i] = 1 if p_1_given_x >= 0.5 else 0
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 10}
        self.clf = MLPClassifier(solver='sgd', alpha=0, activation='logistic',
                    batch_size=1,max_iter=100,
                    learning_rate='constant',learning_rate_init=0.01,
                    shuffle=True,
                    hidden_layer_sizes=(16,), random_state=1)
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None
        
        # For 3 hidden layers
        self.w3 = None
        self.w2 = None
        self.w1 = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(inputs, self.w_input)) 
        a_hidden = np.asmatrix(a_hidden)        
        
        # output activations
        a_output = self.transfer(np.dot(a_hidden, self.w_output))        

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """
        #
        #   x: input; y: output        
        hidden, output = self.feedforward(x) 
        
        
        ### YOUR CODE HERE
        # Find derivative of the loss function regarding to weights1 and weights2
        #nabla_output = np.dot(hidden.T, (2*(y - output) * self.dtransfer(output)))        
        #nabla_output = np.dot(hidden.T, ((y - output) ))        
        
        # part1: 1*1
        part1 = y - output                
        # nabla_output: 4*1
        nabla_output = np.dot(hidden.T,part1)
        
        
        #nabla_input = np.dot(x.T,  (np.dot(2*(y - output) * self.dtransfer(output), self.w_output.T) * self.dtransfer(hidden)))
        #nabla_input = np.dot(x.T,  (np.dot((y - output), self.w_output.T) * output*(1-output)))
        
        # part2: 4*1
        part2 = np.dot(np.dot(np.dot(self.w_output, part1), hidden), (1-hidden).T)
        # nabla_input: 9*4
        nabla_input = np.dot(np.asmatrix(x).T,part2.T)
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    
    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        # Remove column of ones
        Xtrain = np.delete(Xtrain,Xtrain.shape[1]-1,1)
        
        # Testing with scikit
        #self.clf.fit(Xtrain,ytrain)
        
        # size w2: 9*nh
        self.w_input = np.zeros((Xtrain.shape[1],self.params["nh"]))
        # size w1: nh*1
        self.w_output = np.zeros((self.params["nh"] , 1))
        
        ### MY CODE HERE
        numEpochs = self.params["epochs"]
        stepsize = self.params["stepsize"]
        
        # Neural network with Stochastic gradient descent
        for i in range(numEpochs):
            
            # Shuffle Xtrain and Ytrain
            Xtrain, ytrain = shuffleData(Xtrain, ytrain)            
            
            for j in range(Xtrain.shape[0]):                
                # Calculate feed forward first
                hidden, output =  self.feedforward(Xtrain[j,:])
                #Calculate (nabla_input, nabla_output)                
                nabla_input, nabla_output = self.backprop(Xtrain[j,:], ytrain[j])
                self.w_input = self.w_input - stepsize*nabla_input
                self.w_output = self.w_output - stepsize*nabla_output
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """        
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        Xtest = np.delete(Xtest,Xtest.shape[1]-1,1)
        hidden, output = self.feedforward(Xtest)
        ytest = np.where(output >= 0.5, 1, 0)
        
        # Testing with scikit
        #ytest= self.clf.predict(Xtest)
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest
    
    def feedforward3(self, inputs):
        """
        Returns the output of the current neural network for the given input
        2 hidden layers
        """
        # 2 hidden activations
        a_layer3 = self.transfer(np.dot(inputs, self.w3)) 
        a_layer3 = np.asmatrix(a_layer3)        
        a_layer2 = self.transfer(np.dot(a_layer3, self.w2))                 
        # output activations
        a_output = self.transfer(np.dot(a_layer2, self.w1))        

        return (a_layer3,a_layer2, a_output)

    def backprop3(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """
        #
        #   x: input; y: output        
        layer3, layer2, output = self.feedforward(x) 
                
        # part1: 1*1
        part1 = y - output                
        # nabla_output: 4*1 - for updating w1
        nabla_output = np.dot(layer2.T,part1)
                
        # part2: 4*1 -- layer2 - for updating w2
        part2 = np.dot(np.dot(self.w1, part1), self.dtransfer(layer2))
        # nabla_input: 9*4
        nabla_layer2 = np.dot(layer3.T,part2)
        
        # part2: 4*1 -- layer1 - for updating w3
        part3 = np.dot(np.dot(self.w2, part2), self.dtransfer(x,self.w3))
        # nabla_input: 9*4
        nabla_layer3 = np.dot(np.asmatrix(x).T,part3)
        ### END YOUR CODE

        assert nabla_layer3.shape == self.w3.shape
        assert nabla_layer2.shape == self.w2.shape
        assert nabla_output.shape == self.w1.shape
        return (nabla_layer3,nabla_layer2, nabla_output)

class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.k = 100
        self.centers = []
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None'}
        self.reset(parameters)
        
    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = []
        

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None

        ### YOUR CODE HERE
        # Kernel linear
        k = self.k
        K = Xtrain[:k,:]
        self.centers = K
        numsamples = Xtrain.shape[0]
        Ktrain = np.zeros((numsamples,k))
        
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        ### END YOUR CODE
        self.weights = np.random.rand(Ktrain.shape[1],)
        
        if(self.params["kernel"] == "linear"):                   
            Ktrain = np.dot(Xtrain,K.T)
        
        elif(self.params["kernel"] == "hamming"):
            for i in range(Xtrain.shape[0]):
                for j in range(K.shape[0]):
                    hamming_d = 0
                    for d in range(Xtrain.shape[1]):
                        hamming_d += sum(c1 != c2 for c1, c2 in zip(str(Xtrain[i,d]), str(K[j,d])))                        
                            
                    # Ktrain data = nxk matrix
                    Ktrain[i,j] = hamming_d
                
        #self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Ktrain.T,Ktrain)/numsamples,self.params['regwgt']*np.identity(Ktrain.shape[1]))), Ktrain.T),yt)/numsamples
        
        print(Ktrain.shape)
        ### YOUR CODE HERE
        numEpochs = 1000
        stepsize = 0.01
        
        for i in range(numEpochs):
            
            # Shuffle Xtrain and Ytrain
            Ktrain, ytrain = shuffleData(Ktrain, ytrain)
            
            for j in range(Xtrain.shape[1]):                
                #Calculate g
                g1 = utils.sigmoid(np.dot(Ktrain[j,:].T,self.weights)) - ytrain[j]                
                g2= Ktrain[j,:]
                g = np.dot(g1,g2)
                self.weights = self.weights - stepsize*g       
#        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # implement predict functions
    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int) 
        
        K = self.centers
        numsamples = Xtest.shape[0]
        Ktest = np.zeros((numsamples,self.k))
        if(self.params["kernel"] == "linear"):    
            print('Linear is run')
            Ktest = np.dot(Xtest,K.T)
        
        elif(self.params["kernel"] == "hamming"):
            print('hamming is run')
            for i in range(numsamples):
                for j in range(K.shape[0]):
                    hamming_d = 0
                    for d in range(K.shape[1]):
                        hamming_d += sum(c1 != c2 for c1, c2 in zip(str(Xtest[i,d]), str(K[j,d])))                        
                            
                    # Ktrain data = nxk matrix
                    Ktest[i,j] = hamming_d
        
        
        ytest = np.dot(Ktest, self.weights)
        ytest[ytest >= 0] = 1
        ytest[ytest < 0] = 0

        assert len(ytest) == Xtest.shape[0]
        return ytest


# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()


# Function to shuffle data
def shuffleData(Xtrain, ytrain):
    numsamples = Xtrain.shape[0]
    a = list(range(numsamples))    
    np.random.shuffle(a)
    Xtrain = Xtrain[a]
    ytrain = ytrain[a]
    return Xtrain, ytrain
    

# Function to plot figure
def plotFigure(Xdata, Ydata, Xlabel, Ylabel, Title):
    plt.figure()
    plt.plot(Xdata, Ydata)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
        
def splitKeepRatio():
    a = 1
    