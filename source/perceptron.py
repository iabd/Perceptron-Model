# 
#  Created by Abhinav Dwivedi on 25/01/2019.      
#  Copyright © 2019 Abhinav Dwivedi. All rights reserved.                                                                                                                      
# 

import pdb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, hiddenUnits, randomSeed, activation, weightInitialization, epochs, ETA, ALPHA, LAMBDA,loss="MEE", regression=False):
        self.ETA = ETA
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.hiddenUnits = hiddenUnits
        self.activation = activation
        self.lossFunction =loss
        self.weightInitialization = weightInitialization
        self.regression = regression
        self.labelThreshold = 0.5
        self.epochs = epochs
        self.randomSeed=randomSeed
        self.validationAccuracies="The model is NOT trained with validation"
        self.validationLosses="The model is NOT trained with validation"

         
    #@property
    def createModel(self):
        """Returns the structure of the network in form of a dict."""
        np.random.seed(self.randomSeed)    
    # thanks to https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
        if self.weightInitialization=='xav':
            return{
                'Wih':np.random.randn(self.inputUnits,self.hiddenUnits)*np.sqrt(1/self.hiddenUnits),
                'bi':np.random.randn(1,self.hiddenUnits),
                'Who':np.random.randn(self.hiddenUnits,self.outputUnits)*np.sqrt(1/self.hiddenUnits),
                'bh':np.random.randn(1,self.outputUnits)
            }
        elif self.weightInitialization=='zero':
            return{
                'Wih':np.zeros((self.inputUnits,self.hiddenUnits)),
                'Who':np.zeros((self.hiddenUnits,self.outputUnits)),
                'bi':np.zeros((1,self.hiddenUnits)),
                'bh':np.zeros((1,self.outputUnits))
            }
        elif self.weightInitialization=='type1':
            return{
                'Wih':np.random.randn(self.inputUnits,self.hiddenUnits)*np.sqrt(2/self.hiddenUnits),
                'Who':np.random.randn(self.hiddenUnits,self.outputUnits)*np.sqrt(2/self.hiddenUnits),
                'bi':np.random.randn(1,self.hiddenUnits),
                'bh':np.random.randn(1,self.outputUnits)
            }
        elif self.weightInitialization=='type2':
            return{
                'Wih':np.random.randn(self.inputUnits,self.hiddenUnits)*np.sqrt(2/(self.hiddenUnits+self.inputUnits)),
                'Who':np.random.randn(self.hiddenUnits,self.outputUnits)*np.sqrt(2/(self.hiddenUnits+self.outputUnits)),
                'bi':np.random.randn(1,self.hiddenUnits),
                'bh':np.random.randn(1,self.outputUnits)
            }
        else:
            raise NameError("Invalid type of weightInitialization selected")

    def activationFunction(self, input_):
        """Returns the output of given matrix by passing it to the activation function. With the option to use one of three activation functions(sigmoid - 'sigm', Rectified Linear Units - 'relu', Hyperbolic tangent-'tanh', it returns the one specified by instance variable "activation"."""
        if self.activation == 'sigm':
            return 1 / (1 + np.exp(-input_))
        elif self.activation == 'relu':
            return np.maximum(input_, 0)

        elif self.activation == 'tanh':
            return np.tanh(input_)
        else:
            raise NameError("Invalid activation function provided. Please make sure it's 'sigm' or 'relu' or 'tanh")

    def activationDerivative(self, input_):
        """Returns the derivative of the activation function based on parameter "activation"."""
        if self.activation == 'sigm':
            return input_ * (1 - input_)
        if self.activation == 'relu':
            return np.greater(input_, 0)
        if self.activation == 'tanh':
            return 1 - input_ ** 2

    def getLoss(self, yTrue, yPred):
        """Input: yTrue as the true values/labels, yPred as the predicted values by model;
returns the loss. Two loss functions are available. Specifiy the parameter 'loss' while initializing the class to use whichever. """
        if self.lossFunction=="MSE":
            return np.mean(np.square(yTrue - yPred))
        else:
 #       elif self.lossFunction=="MEE":

            return np.mean(np.sqrt(np.sum(np.square(yTrue - yPred), axis=1)))
#        else:
#            raise NameError("Invalid loss function provided. Please make sure it's 'MEE' or 'MSE'")

    def ceilAndFloor(self, input_):
        """If the values are above threshold, sets to 1, otherwise 0. Returns the given list/array of values as numpy array."""
        return np.asarray([0 if i < self.labelThreshold else 1 for i in input_])

    def getAccuracy(self, yTrue, yPred):
        """Returns the accuracy when provided the true and predicted values."""
        if not self.regression:
            yPred=self.ceilAndFloor(yPred)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_pred=yPred, y_true=yTrue)

    def scoreTraining(self, yTrue, yPred, acc=False, loss=False):
        """Input: true and predicted values from the model. Returns the accuracy and/or loss as asked for."""
        #yPred=self.ceilAndFloor(yPred)
        if acc and loss:
            return self.getAccuracy(yTrue, yPred), self.getLoss(yTrue, yPred)
        elif acc:
            return self.getAccuracy(yTrue, yPred)
        elif loss:
            return self.getLoss(yTrue, yPred)
        else:
            raise ValueError("Please specify what to score, 'acc' for accuracy, 'loss' for loss. Set them to True")

    def forwardProp(self, dataMatrix):
        """passes the input matrix via hidden layer to output layer of the network by doing all the computations necessary. Returns the output of hidden layer and output layer."""
        ih_ = np.dot(dataMatrix, self.model['Wih']) + self.model['bi'] #W*X+b
        hh_ = self.activationFunction(ih_) #𝝈
        ho_ = np.dot(hh_, self.model['Who']) + self.model['bh']
        if self.regression:
            return hh_, ho_
        oo_ = self.activationFunction(ho_)
        return hh_, oo_

    def backProp(self, labelMatrix, hh_, oo_):
        """Backpropagation function"""
        difference = labelMatrix - oo_
        if self.regression:
            from sklearn.preprocessing import normalize
            deltaOutput_=normalize(difference, axis=1, norm='l1') #output doesn't passes through nonlinear function for regression

        else:
            deriv = self.activationDerivative(oo_)
            deltaOutput_ = difference * deriv

        deriv = self.activationDerivative(hh_)
        deltaHidden_ = deltaOutput_.dot(self.model['Who'].T) * deriv
        return deltaOutput_, deltaHidden_
    
    def updateWeights(self, dataMatrix, hh_, deltaOutput_, deltaHidden_, prevDeltaWho_, prevDeltaWih_):
        """update the weights"""
        deltaWho_ = hh_.T.dot(deltaOutput_) * self.ETA   
        otherUpdatesWho = self.ETA * self.model['Who'] * (-self.LAMBDA) + self.ALPHA * prevDeltaWho_ #learningrate factor - regularization facor + momentum factor
        deltaWih_ = dataMatrix.T.dot(deltaHidden_) * self.ETA
        otherUpdatesWih = self.ETA * self.model['Wih'] * (-self.LAMBDA) + self.ALPHA * prevDeltaWih_
        if self.regression:
            deltaWho_=deltaWho_/dataMatrix.shape[0]
            deltaWih_ = deltaWih_ / dataMatrix.shape[0]
        #time for updates
        self.model['Who'] += deltaWho_ + otherUpdatesWho
        self.model['Wih'] += deltaWih_ + otherUpdatesWih
        self.model['bh'] += np.sum(deltaOutput_, axis=0, keepdims=True) * self.ETA
        self.model['bi'] += np.sum(deltaHidden_, axis=0, keepdims=True) * self.ETA
        if self.regression:
            self.model['bh']/=dataMatrix.shape[0]
            self.model['bi']/=dataMatrix.shape[0]

        return deltaWho_, deltaWih_

        
    def predict(self, dataMatrix, labels=None, acc_=False, fromVal=False):
        """predict the test/validation dataset
        to get accuracy as well, set acc_ to true
        to get the losses, pass true labels and result of this function to getLoss() function"""
        uselessValue, result=self.forwardProp(dataMatrix)
        if acc_:
            accuracies=[]
            for i in range(dataMatrix.shape[0]):
                assert labels is not None, "true values (as labels) must be provided for to calculate accuracy"
                accuracies.append(self.scoreTraining(labels[i], result[i], acc=True))
            return result, np.sum(accuracies)/dataMatrix.shape[0]
        if fromVal or self.regression:
            return result
        else:
            return self.ceilAndFloor(result)
    

    def fit(self, features, labels, validationFeatures=None, validationLabels=None, realTimePlotting=False):
        """train/fit"""
        self.inputUnits = features.shape[1]
        self.outputUnits = labels.shape[1]
        self.model=self.createModel()
        self.accuracies = []
        self.losses = []
        if validationFeatures is not None:
            self.validationAccuracies=[]
            self.validationLosses=[]
        deltaWho = 0
        deltaWih = 0
        for iteration in range(self.epochs):
            print("iteration {}/{}".format(iteration + 1, self.epochs), end="\r")
            hh, oo = self.forwardProp(features)  #results of hidden layer and output layer after forward propagation)
            deltaOutput, deltaHidden = self.backProp(labels, hh, oo)
            prevDeltaWih = deltaWih
            prevDeltaWho = deltaWho
            deltaWho, deltaWih = self.updateWeights(features, hh, deltaOutput, deltaHidden, prevDeltaWho,prevDeltaWih)
            epochLoss = self.scoreTraining(labels, oo, loss=True)
            self.losses.append(epochLoss)
            if not self.regression:
                epochAccuracy = self.scoreTraining(labels, oo, acc=True)
                self.accuracies.append(epochAccuracy)
            
            if validationFeatures is not None:
                assert len(validationFeatures)==len(validationLabels), "Length of validation features and validation labels must always be the same"
                validationResults=self.predict(validationFeatures, acc_=False, fromVal=True)
                self.validationLosses.append(self.scoreTraining(validationLabels, validationResults, loss=True))
                if not self.regression:
                    self.validationAccuracies.append(self.scoreTraining(validationLabels, validationResults, acc=True))
