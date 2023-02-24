import numpy as np
import math

layer_outputs = [4.8, 1.21, 2.385]

# e - mathematical constant, we use E here to match a common coding style
# where constants are uppercased
E = math.e # 2.71828182846

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)

print('exponentiated values: ')
print(exp_values)

# Normalize values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print('Normalized exponentiated values:')
print(norm_values)

print('Sum of normalized values:', sum(norm_values))