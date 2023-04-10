# Binary Logistic Regression - neurons are either 0 or 1
# Example person/not person or indoors/outdoors
# Uses sigmoid activation (squishes range between 0 and 1) instead of softmax
# Uses Binary Cross-Entropy instead of Categorical Cross-Entropy

import numpy as np


# Sigmoid Activation
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs):
        #save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
