# legendary-train
![0116vsm_McCaffreyRLab](https://github.com/tusharxoxoxo/legendary-train/assets/79051850/b2c2d5bd-274e-49d7-8889-8c96823e8216)

![Screenshot 2023-06-08 at 17 58 31](https://github.com/tusharxoxoxo/legendary-train/assets/79051850/89ea6ba0-33cf-493e-9f19-a86596de3ffd)
![Screenshot 2023-06-08 at 18 00 21](https://github.com/tusharxoxoxo/legendary-train/assets/79051850/8d0a41c1-f2c1-4342-a585-a6390b5e22ea)


This is a Python implementation of a neural network with L1 regularization. The neural network is designed for multi-class classification problems. The implementation includes methods for training the network, computing outputs, and evaluating accuracy and mean squared error.

Dependencies
Python 3.9
NumPy
Usage
To use the nn_L1_alt.py module, follow these steps:

Import the required libraries:
python

'''
import numpy as np
import random
import math
'''

Import the nn_L1_alt.py module:

'''
from nn_L1_alt import NeuralNetwork, makeData, showVector, showMatrixPartial
'''

Create an instance of the NeuralNetwork class:

'''
numInput = 2  # number of input nodes
numHidden = 4  # number of hidden nodes
numOutput = 3  # number of output nodes
seed = 0  # random seed for weight initialization
nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
'''

Train the neural network using training data:

'''
# Generate training data
numRows = 1000  # number of training samples
inputsSeed = 0  # random seed for input generation
trainData = makeData(nn, numRows, inputsSeed)
'''

# Train the neural network

'''
maxEpochs = 1000  # maximum number of training epochs
learnRate = 0.01  # learning rate
L1 = True  # enable L1 regularization
lamda = 0.01  # L1 regularization parameter
nn.train(trainData, maxEpochs, learnRate, L1, lamda)
'''

Evaluate the accuracy of the trained neural network on test data:

'''
# Generate test data
testData = makeData(nn, numRows, inputsSeed)

# Evaluate accuracy
accuracy = nn.accuracy(testData)
print("Accuracy:", accuracy)
'''

Methods
NeuralNetwork class
__init__(self, numInput, numHidden, numOutput, seed)
Initializes the neural network with the specified number of input, hidden, and output nodes.
numInput: Number of input nodes.
numHidden: Number of hidden nodes.
numOutput: Number of output nodes.
seed: Random seed for weight initialization.
setWeights(self, weights)
Sets the weights of the neural network.
weights: List or array of weights.
getWeights(self)
Returns the weights of the neural network as a NumPy array.
initializeWeights(self)
Initializes the weights of the neural network with random values.
computeOutputs(self, xValues)
Computes the outputs of the neural network for the given input values.
xValues: Input values as a list or array.
Returns the computed outputs as a NumPy array.
train(self, trainData, maxEpochs, learnRate, L1=False, lamda=0.0)
Trains the neural network using the provided training data.
trainData: Training data as a NumPy array.
maxEpochs: Maximum number of training epochs.
learnRate: Learning rate for weight updates.
L1: Boolean value indicating whether to use L1 regularization (default: False).
lamda: L1 regularization parameter (default: 0.0).
Returns the trained weights as a NumPy array.
accuracy(self, tdata)
Computes the accuracy of the neural network on the given data.
tdata: Test or train data as a NumPy array.
