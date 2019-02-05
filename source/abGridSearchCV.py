#
#  Created by Abhinav Dwivedi on 30/01/2019.
#  Copyright © 2019 Abhinav Dwivedi. All rights reserved.
#

from sklearn.model_selection import train_test_split
from itertools import product
from source.perceptron import perceptron
import numpy as np


def abGridSearchCV(defaultParams, paramGrid, features, labels, validationSplit, winnerCriteria, log=True, topn=5, logToBeReturned=[]):
    """
    Returns the log(if log=True) and best parameters for the 'perceptron' model by evaluating the model on all the combinations of given parameters.
    Usage:

    logOfGridSearch, topParams=abGridSearchCV(defaultParams, paramGrid, features, labels, validationSplit, winnerCriteria, topn)

    defaultParams: dict of default parameters. (This doesn't affect the model selection)
    paramGrid: dict of parameters as keys with list of params to be tested as values.
    features:features
    labels:labels
    validationSplit:ratio of validation set in overall dataset provided
    winnerCriteria: 'meanTrainingLoss' for the minimum training loss overall, 'meanValidationLoss' with minimum mean training with validation loss, meanLosses for mean of meanLoss and meanValidationLoss'
    log:if True, returns the log of all the combinations,
    topn:how many winners
    """
    
    assert winnerCriteria in ["meanLosses", "meanTrainingLoss", "meanValidationLoss"], "This function currently doesn't support the winner criteria provided. Please make sure it's 'meanLosses' or 'meanTrainingLoss', or 'meanValidationLoss'"
    
    listParams=list(paramGrid.values())
    winners=[]
    allCombinations=list(product(*listParams))
    trainData_, validationData_, trainLabels_, validationLabels_=train_test_split(features, labels, test_size=validationSplit)    
    for index, val in enumerate(allCombinations):
        print("                        {}/{}".format(index, len(allCombinations)), end="\r")
        param={}
        for index_ in range(len(paramGrid)):
            param[list(paramGrid.keys())[index_]]=val[index_]
        regressionModel=perceptron(**defaultParams)
        regressionModel.set_params(**param)
        regressionModel.fit(trainData_, trainLabels_, validationData_, validationLabels_)
        meanLoss=np.mean(regressionModel.losses)
        meanValidationLoss=np.mean(regressionModel.validationLosses)
        tempLog={
            'params': param, 
            'meanTrainingLoss':meanLoss, 
            'meanValidationLoss':meanValidationLoss,
            'meanLosses':(meanLoss+meanValidationLoss)/2
        }
        if log:
            logToBeReturned.append(tempLog)
            
        if len(winners)<topn:
            winners.append(tempLog)
        else:
            winners = sorted(winners, key=lambda k: k[winnerCriteria])
            if tempLog[winnerCriteria]<winners[-1][winnerCriteria]:
                winners[-1]=tempLog
            
    if log:
        return logToBeReturned, winners
    else:
        return winners
