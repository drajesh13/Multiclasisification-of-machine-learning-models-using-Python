# Import necessary modules
from joblib.numpy_pickle_utils import xrange
from numpy import *

# Define the NeuralNet class
class NeuralNet(object): 
    def __init__(self): 
        # Initialize the neural network with random weights between -1 and 1
        random.seed(1) 
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Sigmoid activation function
    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x))

    # Derivative of the sigmoid function
    def __sigmoid_derivative(self, x): 
        return x * (1 - x) 

    # Training function
    def train(self, inputs, outputs, training_iterations): 
        for iteration in xrange(training_iterations): 
            # Forward pass
            output = self.learn(inputs) 
            # Calculate the error
            error = outputs - output 

            # Calculate the adjustment to the weights based on the error and sigmoid derivative
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
            # Update the weights
            self.synaptic_weights += factor 

    # Forward pass function
    def learn(self, inputs): 
        # Calculate the output of the neural network using the sigmoid activation function
        return self.__sigmoid(dot(inputs, self.synaptic_weights)) 
