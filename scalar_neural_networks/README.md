
# Neural Networks from Scratch — Scalar Experiments
 
This repository contains three of my early experiments in building neural networks from scratch using Python and scalar mathematics.

The goal of these projects is to understand how neurons, weights, biases, forward propagation, loss calculation, gradients, and learning work internally, without relying on machine learning frameworks.

These are learning experiments built out of curiosity and a desire to understand the mathematics behind neural networks.

## Requirements
python

## Installation

Navigate:

```bash
cd neural-network-experiments/scalar_neural_network
```

## Run Experiments

`neuron_scalar`:
```bash
python3 neuron_scalar.py
```

`neural_network_linear_scalar`:
```bash
python3n neural_network_linear_scalar.py
```

`neural_network_non-linear_scalar`:
```bash
python3 neural_network_non-linear_matrix.py
```
 
## Experiments

### 1. Single Neuron — `neuron_scalar.py`

A single neuron with two inputs, two weights, and a bias.

- Learns to perform addition.
- Uses randomly generated training data.
- Calculates the prediction error using squared error.
- Updates weights and bias using a simple learning rule.
- Tests the trained neuron with a new input.

This was my starting point for understanding how a single artificial neuron learns from data.

### 2. Linear Neural Network — `neural_network_linear_scalar.py`

A neural network consisting of three hidden neurons and one output neuron, implemented using scalar operations.

- Learns to perform addition.
- Implements forward propagation manually.
- Calculates gradients for weights and biases.
- Updates parameters using gradient descent.
- Uses no activation function between the hidden neurons and the output neuron.

This experiment helped me explore how multiple neurons work together and how gradients can be calculated manually.

### 3. Non-Linear Neural Network — `neural_network_non-linear_scalar.py`

A neural network with seven hidden neurons and one output neuron, using the Leaky ReLU activation function.

- Learns to approximate absolute subtraction: `abs(x1 - x2)`.
- Implements Leaky ReLU and its derivative.
- Calculates gradients manually using the chain rule.
- Updates weights and biases through gradient descent.
- Tests the trained network on a new input.

This experiment introduced me to activation functions and the role of the chain rule in training a nonlinear neural network.

NOTE - The experiments generate training data and initialize parameters within their scripts. Results may vary between runs when random initialization or random training data are used.

Thank you :D
