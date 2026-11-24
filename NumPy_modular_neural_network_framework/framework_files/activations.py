import numpy as np

# identity
def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

# ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Leaky ReLU
def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.square(tanh(x))

# softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)

    diagonal = np.eye(s.shape[-1]) * s[..., None, :]
    outer = s[..., :, None] * s[..., None, :]

    return diagonal - outer