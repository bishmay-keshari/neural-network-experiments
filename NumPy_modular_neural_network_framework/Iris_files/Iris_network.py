import numpy as np

from framework_files import activations
from framework_files.numpy_framework import Layer, Network, Early_Stopping, Exponential_Moving_Average
from framework_files.losses import categorical_cross_entropy, categorical_cross_entropy_derivative
from framework_files.optimizers import Adam
from framework_files.data_preparation import Iris

np.random.seed(7)

# LAYERS
network = Network([
    Layer(4, 8, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(8, 16, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(16, 8, activations.tanh, activations.tanh_derivative, weight_init="xavier"),
    Layer(8, 3, activations.softmax, activations.softmax_derivative, weight_init="xavier")
])

# DATASET
training_data, validation_data, test_data = Iris.Iris_data()

train_input, train_target = training_data[0]
validation_input, validation_target = validation_data[0]
test_input, test_target = test_data[0]

# TRAINING
epochs = 10000
epoch_print_interval = 100

learning_rate = 0.001
batch_size = 16

loss_function = categorical_cross_entropy
loss_function_derivative = categorical_cross_entropy_derivative

early_stopping = Early_Stopping(patience=100, min_delta=1e-4)
exponential_moving_average = Exponential_Moving_Average(network, decay=0.99)

optimizer = Adam(beta1=0.9, beta2=0.999, epsilon=1e-8)

network.train(training_data, epochs, epoch_print_interval, learning_rate, batch_size, loss_function, loss_function_derivative, optimizer, validation_data=validation_data, early_stopping=early_stopping, exponential_moving_average=exponential_moving_average)
train_output = network.forward(train_input)
train_loss = categorical_cross_entropy(train_output, train_target)

print("Train_loss:", train_loss)

# VALIDATION
validation_output = network.forward(validation_input)
validation_loss = categorical_cross_entropy(validation_output, validation_target)

print("Validation_loss:", validation_loss)

# TESTING
test_output = network.forward(test_input)
test_loss = categorical_cross_entropy(test_output, test_target)

print("Test_Loss:", test_loss)

# ACCURACY
predicted_classes_train = np.argmax(train_output, axis=1)
train_accuracy = np.mean(predicted_classes_train == np.argmax(train_target, axis=1))

predicted_classes_validation = np.argmax(validation_output, axis=1)
validation_accuracy = np.mean(predicted_classes_validation == np.argmax(validation_target, axis=1))

predicted_classes_test = np.argmax(test_output, axis=1)
test_accuracy = np.mean(predicted_classes_test == np.argmax(test_target, axis=1))

print("Train Accuracy:", train_accuracy)
print("Validation Accuracy:", validation_accuracy)
print("Test Accuracy:", test_accuracy)

# SAVE SYSTEM
np.save("original_output_Iris.npy", test_output)
network.save("Iris_network.npz")
print("Network Saved !!!")