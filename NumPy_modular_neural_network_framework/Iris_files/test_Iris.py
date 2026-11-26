import numpy as np

from framework_files import activations
from framework_files.numpy_framework import Layer, Network
from framework_files.data_preparation import Iris

network = Network([
    Layer(4, 8, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(8, 16, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(16, 8, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(8, 3, activations.softmax, activations.softmax_derivative, weight_init="xavier")
])

training_data, validation_data, test_data = Iris.Iris_data()
test_input, test_target = test_data[0]
network.load("Iris_network.npz")

output = network.forward(test_input)
print(f"Output: {output}")

original_output = np.load("original_output_Iris.npy")
print("Output Match:", np.allclose(original_output, output))