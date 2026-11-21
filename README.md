# Neural-Network-Experiments
These are my early projects about neural network. This is basically meant for understanding neural network from the ground up. 


NO AI WAS USED TO BUILD MY PROJECTS. THE PROJECTS ARE BUILT BY ME AND I DESIGNED THE ARCHITECTURE.


Check Below For more information.

## Installation
Clone the repository:

```bash
git clone https://github.com/bishmay-keshari/neural-network-experiments.git
```

## Projects

### Scalar_Neural_Network

This is based on building neural networks without any dependencies, in pure python.
It has currently 3 individual experiments.

`neuron_scalar.py` is a single scalar neuron.

`neural_network-linear_scalar.py` are multiple neurons working together but in linear form, without activation function.

`neural_network_non-linear_scalar.py` are multiple neurons working together with Leaky ReLU action function.

for more details check out the README file in scalar_neural_network.

### Matrix_Neural_Network

This is based on building neural networks using matrices with NumPy only.
It has currently 4 individual experiments.

`neuron_matrix.py` is a single neuron written in matrix by NumPy.

`neural_network-linear_matrix.py` are multiple neurons working together but in linear form, without activation function.

`neural_network_non-linear_matrix.py` are multiple neurons working together with Leaky ReLU activation function.

`multi_layer_neural_network.py` has multiple hidden layers built on NumPy only.

for more details check out the README file in matrix_neural_network.

### Modular_Neural_Network_Framework

This is a reusable modular neural network built upon NumPy only.
Unlike previous files which are individual this is system of files working together.

`framework_files` has numpy_framework, activations, losses, optimizers.
"numpy_framework" has class Layer, class Network which act as backbone on which different architecture like XOR, Iris, MNIST models are built.
EMA for parameters and Early Stopping are also implemented as optional features.

The Network class can also save and load trained parameters.

"activations" has all the activation functions like ReLU, Tanh, Softmax etc.

"losses" includes all the loss functions such as MSE, BCE, CCE etc.

"optimizers" contains vanilla SGD, SGD with Momentum and Adam.

"data_preparation" makes the dataset suitable for NumPy matrices reading the iris.data file with python inbuilt CSV or reading MNIST raw idx binary format with struct, and then shuffle them and split the data, then do normalization in NumPy only.

And also I built XOR network, Iris network, MNIST network based on this framework.
You can get more information on the README file in modular_neural_network_framework.

