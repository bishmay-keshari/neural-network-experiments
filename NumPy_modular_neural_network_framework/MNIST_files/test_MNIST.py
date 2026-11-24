import numpy as np

from framework_files import activations
from framework_files.numpy_framework import Layer, Network
from framework_files.data_preparation import MNIST

network = Network([
    Layer(784, 128, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(128, 64, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(64, 32, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(32, 10, activations.softmax, activations.softmax_derivative, weight_init="xavier")
])

training_data, validation_data, test_data = MNIST.MNIST_data()
test_input, test_target = test_data[0]
network.load("MNIST_network.npz")

output = network.forward(test_input)
print(f"Output: {output}")

original_output = np.load("original_output_MNIST.npy")
print("Output Match:", np.allclose(original_output, output))