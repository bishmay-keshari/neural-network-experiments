import numpy as np

from framework_files import activations
from framework_files.numpy_framework import Layer, Network

network = Network([
    Layer(2, 4, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(4, 4, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(4, 1, activations.sigmoid, activations.sigmoid_derivative, weight_init="xavier")
])

input_data  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
network.load("XOR_network.npz")

output = network.forward(input_data)
print(f"Output: {output}")

original_output = np.load("original_output_XOR.npy")
print("Output Match:", np.allclose(original_output, output))