import numpy as np

# MAE {mean_absolute_error}

def mean_absolute_error(prediction, target):
    return np.mean(np.abs(target - prediction))

def mean_absolute_error_derivative(prediction, target):
    return np.sign(prediction - target) / prediction.size

# MSE {mean_squared_error}

def mean_squared_error(prediction, target):
    return 0.5 * np.mean((target - prediction) ** 2)

def mean_squared_error_derivative(prediction, target):
    return (prediction - target) / prediction.size

# BCE {binary_cross_entropy}

def binary_cross_entropy(prediction, target):
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1-epsilon)
    return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

def binary_cross_entropy_derivative(prediction, target):
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return ((1 - target) / (1 - prediction) - target / prediction) / prediction.size

# CCE {categorical_cross_entropy}
def categorical_cross_entropy(prediction, target):
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return -np.mean(np.sum(target * np.log(prediction), axis=-1))

def categorical_cross_entropy_derivative(prediction, target):
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return (-target / prediction) / prediction.shape[0]