# Dropout Layer - Disables some neurons while others pass through unchange
# Similar to Regularization - Prevent neural network from becoming too dependent on
# any neuron

# Dropout helps with co-adoption and noise
# More neurons working together mean that the model can learn more complex functions

# Bernoulli distribution
# binary (discrete) probability distribution

import random
import numpy as np

# dropout_rate = 0.5
# # Example output containing 10 values
# example_output = [0.27, -1.03, 0.67, 0.99, 0.05,
#                   -0.37, -2.01, 1.13, -0.07, 0.73]
#
# # Repeat as long as necessary
# while True:
#
#     # Randomly choose index and set value to 0
#     index = random.randint(0, len(example_output) - 1)
#     example_output[index] = 0
#
#     # We might set an index that already is zeroed
#     # There are different ways of overcoming this problem
#     # for simplicity we count values that are exactly 0
#     # while its extremely rare in real model that weights
#     # are exactly 0, this is not the best method for sure
#     dropped_out = 0
#     for value in example_output:
#         if value == 0:
#             dropped_out += 1
#
#     # If required number of outputs is zeroed - leave the loop
#     if dropped_out / len(example_output) >= dropout_rate:
#         break

# print(example_output)

# array = np.random.binomial(2, 0.5, size=10)
# print(array)

# dropout_rate = 0.2
# print(np.random.binomial(1, 1-dropout_rate, size=5))

# dropout_rate = 0.3
# example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
#                            -0.37, -2.01, 1.13, -0.07, 0.73])
#
# example_output *= np.random.binomial(1, 1-dropout_rate, example_output.shape)
#
# print(example_output)

dropout_rate = 0.2
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])
print(f'sum initial {sum(example_output)}')

sums = []
for i in range(10000):

    example_output2 = example_output * \
        np.random.binomial(1, 1-dropout_rate, example_output.shape) / \
        (1 - dropout_rate)
    sums.append(sum(example_output2))

print(f'mean sum: {np.mean(sums)}')

