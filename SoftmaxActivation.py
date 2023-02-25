import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#Dense Layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initalize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Forward Pass
    def forward(self, inputs):
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output vales
dense1 = Layer_Dense(2,3)

# Create ReLU activation (to be used with dense layer):
activation1 = Activation_ReLU()

# Create Dense layer with 3 input features (outputs from previous layer) and 3 output vales
dense2 = Layer_Dense(3,3)

# Create Softmax activation (to be used with dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function
# Takes in output from previous layer
activation1.forward(dense1.output)

# Make a forward pass of our training data through this layer
# Takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Forward pass through activation function
# Takes in output of second dense layer here
activation2.forward(dense2.output)

print(activation2.output[:5])