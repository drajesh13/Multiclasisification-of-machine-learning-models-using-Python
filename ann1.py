# from numpy import exp, array, random, dot
# import numpy as np

# class NeuralNet(object): 
#     def __init__(self, input_nodes, output_nodes): 
#         random.seed(1) 
#         self.synaptic_weights = 2 * random.random((input_nodes, output_nodes)) - 1

#     # Softmax activation function
#     def __softmax(self, x):
#         e_x = exp(x - np.max(x))
#         return e_x / e_x.sum(axis=1, keepdims=True)

#     # Cross-entropy loss
#     def __cross_entropy(self, output, y_target):
#         m = y_target.shape[0]
#         log_likelihood = -np.log(output[range(m), y_target])
#         loss = np.sum(log_likelihood) / m
#         return loss

#     # Training function
#     def train(self, inputs, outputs, training_iterations): 
#         for iteration in range(training_iterations): 
#             # Forward pass
#             output = self.learn(inputs) 
#             # Calculate the error
#             error = self.__cross_entropy(output, outputs)

#             # Calculate the adjustment to the weights based on the error and softmax derivative
#             factor = dot(inputs.T, error * (output * (1 - output))) 
#             # Update the weights
#             self.synaptic_weights += factor 

#     # Forward pass function
#     def learn(self, inputs): 
#         # Calculate the output of the neural network using the softmax activation function
#         return self.__softmax(dot(inputs, self.synaptic_weights))

#     # Prediction function
#     def predict(self, inputs):
#         # Forward pass through the network
#         output = self.learn(inputs)
#         # Return the class with the highest probability
#         return np.argmax(output, axis=1)
    
# X_train = array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]]) 
# y_train = array([0, 1, 2, 1, 0])

# # Initialize the model with the appropriate number of input and output nodes
# ann_model = NeuralNet(input_nodes=2, output_nodes=3)
# ann_model.train(X_train, y_train, training_iterations=10000)

# X_test = array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]) 
# predictions = ann_model.predict(X_test) 
# print(predictions)

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(output, y_target):
    m = y_target.shape[0]
    log_likelihood = -np.log(output[range(m), y_target])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X, y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad/m
    return grad

class NeuralNetwork:
    def __init__(self, x, y, learning_rate=0.1):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4, len(np.unique(y)))                 
        self.y          = y
        self.output     = np.zeros(y.shape)
        self.learning_rate = learning_rate

    def feedforward(self):
        self.layer1 = softmax(np.dot(self.input, self.weights1))
        self.output = softmax(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (delta_cross_entropy(self.output, self.y)))
        d_weights1 = np.dot(self.input.T,  (np.dot(delta_cross_entropy(self.output, self.y), self.weights2.T)))

        self.weights1 -= self.learning_rate * d_weights1
        self.weights2 -= self.learning_rate * d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
    
    def predict(self, X):
        self.input = X
        self.feedforward()
        return np.argmax(self.output, axis=1)

# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])
# y = np.array([[1],
#               [2],
#               [1],
#               [0]])

# nn = NeuralNetwork(X, y, learning_rate=0.01)

# for i in range(5000):  # Increase the number of training iterations
#     nn.feedforward()
#     nn.backprop()

# print(nn.output)

# # Create new data
# X_new = np.array([[0, 0, 1],
#                   [0, 1, 1],
#                   [1, 0, 1],                  
#                   [1, 1, 1]])

# # Make predictions
# predictions = nn.predict(X_new)
# print(predictions)