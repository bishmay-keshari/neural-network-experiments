# Modular_Neural_Network_Framework-Scratch

This repository contains my experiments in building a modular neural network framework from scratch using Python and NumPy.

The framework is built using NumPy and implements the fundamental operations involved in neural network.

The project also explores how the same framework can be used to solve different problems, from simple logical operations to multiclass classification and handwritten digit recognition.

## Technologies
 
- Python 3
- NumPy
- Python standard libraries- 'csv', 'struct', 'pathlib'

The neural network calculations are implemented using NumPy without relying on deep-learning frameworks.

## Requirements

Python 3 and NumPy are required.

### Install NumPy with:

```bash
python3 -m pip install numpy
```

## Installation

Clone the repository:

```bash
git clone https://github.com/bishmay-keshari/neural-network-experiments.git
```

Navigate to the project directory:

```bash
cd neural-network-experiments/modular_neural_network_framework
```

Install it in editor mode:

```bash
pip install -e .
```
The project uses `pyproject.toml` to configure its package installation.

The editable installation allows Python to locate the framework modules when running the experiment scripts.

## Running the Experiments

They can be executed from the project root using the terminal:

```bash
python -m XOR_files.XOR_network
```

```bash
python -m Iris_files.Iris_network
```

```bash
python -m MNIST_files.MNIST_network
```

## Datasets

The data preparation module currently supports:

- **XOR:** Generated directly inside the experiment.
- **Iris:** Requires the `iris.data` dataset file.
- **MNIST:** Requires the four original IDX dataset files:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

The datasets are not included in this repository and must be obtained separately.

Dataset paths in `data_preparation.py` currently need to be configured according to the local directory structure.

## Experiments

### 1. XOR Binary Classification

A neural network designed to learn the XOR logical operation.

- Uses a multilayer neural network architecture.
- Implements Tanh activation functions in the hidden layers.
- Uses Sigmoid activation in the output layer.
- Uses Binary Cross-Entropy (BCE) as the loss function.
- Implements SGD with momentum.
- Uses mini-batch training.

This experiment explores how a neural network can learn a nonlinear logical relationship.

### 2. Iris Flower Classification

A neural network trained to classify three different species of Iris flowers.

- Uses a multilayer neural network architecture.
- Uses Tanh activation in the hidden layers.
- Uses Softmax activation in the output layer.
- Uses Categorical Cross-Entropy (CCE) as the loss function.
- Implements the Adam optimizer.
- Uses Early Stopping to monitor validation loss.
- Implements Exponential Moving Average (EMA) for parameter averaging.
- Evaluates the network using training, validation, and test datasets.

This experiment explores multiclass classification and different techniques used during neural network training.

### 3. MNIST Handwritten Digit Classification

A neural network designed to classify handwritten digits from the MNIST dataset.

- Uses a multilayer neural network architecture.
- Uses Tanh activation in the hidden layers.
- Uses Softmax activation in the output layer.
- Uses Categorical Cross-Entropy (CCE) as the loss function.
- Implements the Adam optimizer.
- Uses mini-batch training.
- Implements Early Stopping and Exponential Moving Average (EMA).
- Evaluates the network using training, validation, and test datasets.

This experiment explores how the framework can be applied to image classification.

### 1. Neural Network Framework — `numpy_framework.py`

Contains the main components of the neural network:

- **Layer** — handles weights, biases, activation functions, forward propagation, and backpropagation.
- **Network** — connects multiple layers and manages forward propagation, backpropagation, training, and parameter updates.
- **Early Stopping** — monitors validation loss and restores the best recorded network parameters.
- **Exponential Moving Average (EMA)** — maintains moving averages of network parameters.

The framework also supports saving and loading trained network parameters.

### 2. Activation Functions — `activations.py`

Implements different activation functions and their derivatives:

- Identity
- ReLU
- Leaky ReLU
- Sigmoid
- Tanh
- Softmax

### 3. Loss Functions — `losses.py`

Implements different loss functions and their derivatives:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Binary Cross-Entropy (BCE)
- Categorical Cross-Entropy (CCE)

### 4. Optimizers — `optimizers.py`

Contains different optimization algorithms:

- Vanilla SGD
- SGD with Momentum
- Adam

## Project Structure

The project is organized into separate folders to keep the framework and individual experiments independent.

```text
modular_neural_framework-scratch/
│
├── pyproject.toml
│
├── framework_files/
│   ├── __init__.py
│   ├── numpy_framework.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizers.py
│   └── data_preparation.py
│
├── XOR_files/
│   ├── __init__.py
│   ├── XOR_network.py
│   └── test_XOR.py
│
├── iris_files/
│   ├── __init__.py
│   ├── iris_network.py
│   └── test_iris.py
│
└── MNIST_files/
    ├── __init__.py
    ├── MNIST_network.py
    └── test_MNIST.py
```

Thank you :D
