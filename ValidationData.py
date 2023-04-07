# Hyperparameter tuning should be done on Validation Data
# Not Training Data!!
# Cross-Validation is used on small training datasets that cannot afford any data for validation purposes
# k-fold cross-validation
# Common to loop over different hyperparameter sets, leaving the code to run training multiple times, applying different settings
# each run, and reviewing the results to choose the best set of hyperparameters