import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:

    # Forward Pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Common loss class
class Loss:
    # Calculates the data and regulatization losses
    # given model ouput and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # return loss
        return data_loss

# Cross Entropy Loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # number of samples in a batch
        samples = len(y_pred)

        #Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )

        # Losses
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

# Perform a forward pass through activation function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

# Calculate accuracy from output activation2 and targets
# Calculate valyes along first axis

# Print loss value
print('loss:', loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

# Print Accuracy
print('acc:', accuracy)




# softmax_outputs = np.array([[0.7, 0.1, 0.2],
#                             [0.1, 0.5, 0.4],
#                             [0.02, 0.9, 0.08]])
# class_targets = np.array([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 1, 0]])
#
# # Probabilities for target values
# # Only if categorical labels
# # if len(class_targets.shape) == 1:
# #  correct_cofidences = softmax_outputs[
# #   range(len(softmax_outputs)),
# #   class_targets
# #  ]
# # elif len(class_targets.shape) == 2:
# #  correct_confidence = np.sum(
# #   softmax_outputs*class_targets,
# #   axis=1
# #  )
# #
# # # Losses
# # neg_log = -np.log(correct_confidence)
# #
# # average_loss = np.mean(neg_log)
# # print(average_loss)
#
# # Common loss class
# class Loss:
#
#     # Calculates the data and regularization losses
#     # given model output and ground truth values
#     def calculate(self, output, y):
#
#         # calculate sample loss
#         sample_losses = self.forward(output, y)
#
#         # calculate mean loss
#         data_loss = np.mean(sample_losses)
#
#         # Return loss
#         return data_loss
#
# # Cross-entropy loss
# class Loss_CategoricalCrossentropy(Loss):
#
#     # Forward Pass
#     def forward(self, y_pred, y_true):
#
#         # Number of samples in a batch
#         samples = len(y_pred)
#
#         # Clip data to prevent division by 0
#         # Clip both sides to not drag mean towards any value
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
#
#         # Probabilities for target values
#         # only if categorical labels
#         if len(y_true.shape) == 1:
#             correct_confidences = y_pred_clipped[
#                 range(samples),
#                 y_true
#             ]
#
#         # Mask values - only for one hot encoded labels
#         elif len(y_true.shape) == 2:
#             correct_confidences = np.sum(
#                 y_pred_clipped*y_true,
#                 axis=1
#             )
#
#         # Losses
#         negative_log_likelihoods = -np.log(correct_confidences)
#         return negative_log_likelihoods
#
# loss_function = Loss_CategoricalCrossentropy()
# loss = loss_function.calculate(softmax_outputs, class_targets)
# print(loss)