import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative, weight_init):
        
        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))

        elif weight_init == "he":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)

        else:
            raise ValueError(f"Unknown weight initialization method: {weight_init}")
        
        self.bias = np.zeros((1, output_size))

        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias)

        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def backward(self, output_gradient):
        activation_gradient = self.activation_derivative(self.z)

        if activation_gradient.ndim == output_gradient.ndim + 1:
            delta = np.einsum('bi,bij->bj', output_gradient, activation_gradient)
        else:
            delta = output_gradient * activation_gradient

        weights_gradient = np.dot(self.input_data.T, delta)
        bias_gradient = np.sum(delta, axis=0, keepdims=True)
        input_gradient = np.dot(delta, self.weights.T)

        return input_gradient, weights_gradient, bias_gradient


class Early_Stopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_loss = float("inf")
        self.best_parameters = None
        self.best_epoch = 0

    def check(self, network, validation_loss, epoch, parameters=None):

        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            if parameters is None:
                self.best_parameters = network.get_parameters()
            else:
                self.best_parameters = [(w.copy(), b.copy()) for w, b in parameters]
            self.best_epoch = epoch + 1
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience

    def restore_best_parameters(self, network):
        if self.best_parameters is not None:
            network.set_parameters(self.best_parameters)
            print("Restored best parameters from epoch:", self.best_epoch)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        output = input_data

        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient):
        gradients = []
        gradient = output_gradient

        for layer in reversed(self.layers):
            gradient, weights_gradient, bias_gradient = layer.backward(gradient)
            gradients.append((weights_gradient, bias_gradient))

        gradients.reverse()
        return gradients

    def update(self, gradient, learning_rate, optimizer):
        optimizer.update(self.layers, gradient, learning_rate)

    def evaluate_exponential_moving_average(self, validation_data, loss_function, ema):
        original_parameters = self.get_parameters()

        try:
            self.set_parameters(ema.get_parameters())
            return self.evaluate_early_stopping(validation_data, loss_function)
        finally:
            self.set_parameters(original_parameters)

    def evaluate_early_stopping(self, validation_data, loss_function):
            total_loss = 0.0
            total_samples = 0
        
            for input_data, target in validation_data:
                output = self.forward(input_data)
                loss = loss_function(output, target)
        
                batch_samples = len(input_data)
                total_loss += loss * batch_samples
                total_samples += batch_samples

            if total_samples == 0:
                raise ValueError("validation_data not provided or empty")

            return total_loss / total_samples

    def get_parameters(self):
        parameters = []
        
        for layer in self.layers:
            parameters.append((layer.weights.copy(), layer.bias.copy()))
        return parameters
        
    def set_parameters(self, parameters):
        for layer, (weights, bias) in zip(self.layers, parameters):
            layer.weights = weights.copy()
            layer.bias = bias.copy()
    
    def train(self, training_data ,epochs, epoch_print_interval, learning_rate, batch_size, loss_function, loss_function_derivative, optimizer, validation_data=None, early_stopping=None, exponential_moving_average=None):

        if early_stopping is not None and validation_data is None:
            raise ValueError("Validation data must be provided when using early stopping")

        for epoch in range(epochs):
            total_loss = 0.0
            total_samples = 0
            
            for input_data, target in training_data:
                # Shuffling
                indices = np.random.permutation(len(input_data))
                input_data = input_data[indices]
                target = target[indices]

                # mini-batch
                for start in range(0, len(input_data), batch_size):
                    batch_input = input_data[start:start + batch_size]
                    batch_target = target[start:start + batch_size]

                    batch_samples = len(batch_input)
                    total_samples += batch_samples
                 
                    # forward pass    
                    output = self.forward(batch_input)

                    # loss calculation
                    loss = loss_function(output, batch_target)
                    total_loss += loss * batch_samples
                
                    # backpropagation
                    output_gradient = loss_function_derivative(output, batch_target)
                    gradient = self.backward(output_gradient)

                    #update
                    self.update(gradient, learning_rate, optimizer)

                    # update exponential moving average
                    if exponential_moving_average is not None:
                        exponential_moving_average.update(self)

            average_loss = total_loss / total_samples

            if (epoch + 1) % epoch_print_interval == 0:
                print("Epoch", epoch + 1, "Loss:", average_loss)

            # Early Stopping
            if validation_data is not None:
                if exponential_moving_average is not None:
                    validation_loss = self.evaluate_exponential_moving_average(validation_data, loss_function, exponential_moving_average)
                else:
                    validation_loss = self.evaluate_early_stopping(validation_data, loss_function)

                parameters =(exponential_moving_average.get_parameters()
                            if exponential_moving_average is not None else self.get_parameters())
                
                if early_stopping is not None:
                    if early_stopping.check(self, validation_loss, epoch, parameters):
                        print("Early stopping triggered at epoch", epoch + 1)
                        break
                    
        if early_stopping is not None:            
            early_stopping.restore_best_parameters(self)
        
    def save(self, filename):
        parameters = {}
        for i, layer in enumerate(self.layers):
            parameters[f"weights_{i}"] = layer.weights
            parameters[f"bias_{i}"] = layer.bias

        np.savez(filename, **parameters)

    def load(self, filename):
        parameters = np.load(filename)
        for i, layer in enumerate(self.layers):
            layer.weights = parameters[f"weights_{i}"].copy()
            layer.bias =  parameters[f"bias_{i}"].copy()


class Exponential_Moving_Average:
    def __init__(self, network, decay=0.99):
        self.decay = decay
        self.ema_parameters = network.get_parameters()

    def update(self, network):
        current_parameters = network.get_parameters()

        for i, (weights, bias) in enumerate(current_parameters):
            ema_weights, ema_bias = self.ema_parameters[i]

            ema_weights = self.decay * ema_weights + (1 - self.decay) * weights
            ema_bias = self.decay * ema_bias + (1 - self.decay) * bias

            self.ema_parameters[i] = (ema_weights, ema_bias)

    def get_parameters(self):
        return self.ema_parameters