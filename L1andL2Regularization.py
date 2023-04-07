# Regularization methods are those which reduce generalization error
# L1 and L2 regularization are used to calculate penalty
# Generally it is better to have many neurons contributing to a models output rather than a select few

# L2 regularization is commonly used as it does not affect small parameter values substantially
# L1 regularization is rarely used alone

# Lambda - dictates how much of an impact this regularization penalty carries
# higher value means a more significant penalty

# Regularization Code Notation
# l1w = lambda_l1w * sum(abs(weights))
# l1b = lambda_l1b * sum(abs(biases))
# l2w = lambda_l2w * sum(weights**2)
# l2b = lambda_l2b * sum(biases**2)
# loss = data_loss + l1w + l1b + l2w + l2b


