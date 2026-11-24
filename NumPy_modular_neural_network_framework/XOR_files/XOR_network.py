import numpy as np

from framework_files import activations
from framework_files.numpy_framework import Layer, Network
from framework_files.losses import binary_cross_entropy, binary_cross_entropy_derivative
from framework_files.optimizers import SGD_with_momentum

np.random.seed(23)

# LAYERS
network = Network([
    Layer(2, 4, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(4, 4, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(4, 1, activations.sigmoid, activations.sigmoid_derivative, weight_init="xavier")
])

# TRAINING
input_data  = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
target = np.array([[0.0], [1.0], [1.0], [0.0]],dtype=np.float32)

training_data = [(input_data, target)]

epochs = 10000
epoch_print_interval = 1000

learning_rate = 0.5
batch_size = 2

loss_function = binary_cross_entropy
loss_function_derivative = binary_cross_entropy_derivative

optimizer = SGD_with_momentum(momentum=0.9)

network.train(training_data, epochs, epoch_print_interval, learning_rate, batch_size, loss_function, loss_function_derivative, optimizer)
output = network.forward(input_data)
loss = binary_cross_entropy(output, target)

print("Target:", target, "Output:", output)
print("Loss:", loss)

# ACCURACY
predicted_classes = (output >= 0.5)
accuracy = np.mean(predicted_classes == target)
print("Accuracy:", accuracy)

# SAVE SYSTEM
np.save("original_output_XOR.npy", output)
network.save("XOR_network.npz")
print("Network Saved !!!")