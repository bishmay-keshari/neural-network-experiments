import numpy as np

class vanilla_SGD:
    def __init__(self):
        pass
        
    def update(self, layers, gradient, learning_rate):
        for layer, (weights_gradient, bias_gradient) in zip(layers, gradient):

            # Update parameters
            layer.weights = layer.weights - learning_rate * weights_gradient
            layer.bias = layer.bias - learning_rate * bias_gradient

class SGD_with_momentum:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = None

    def update(self, layers, gradient, learning_rate):
        for layer, (weights_gradient, bias_gradient) in zip(layers, gradient):

            # Update velocity
            if self.velocity is None:

                self.velocity = [
                    (np.zeros_like(layer.weights), np.zeros_like(layer.bias)) for layer in layers]

            layer.weight_velocity = self.momentum * layer.weight_velocity + weights_gradient
            layer.bias_velocity = self.momentum * layer.bias_velocity + bias_gradient

            # Update parameters
            layer.weights = layer.weights - learning_rate * layer.weight_velocity
            layer.bias = layer.bias - learning_rate * layer.bias_velocity

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, layers, gradient, learning_rate):

        if self.m is None and self.v is None:
            self.m = [(np.zeros_like(layer.weights), np.zeros_like(layer.bias)) for layer in layers]
            self.v = [(np.zeros_like(layer.weights), np.zeros_like(layer.bias)) for layer in layers]

        # Increment time step
        self.t += 1

        for i, (layer, (weights_gradient, bias_gradient)) in enumerate(zip(layers, gradient)):
            # Update first moment estimate
            m_weights, m_bias = self.m[i]

            m_weights = (self.beta1 * m_weights) + (1 - self.beta1) * weights_gradient
            m_bias = (self.beta1 * m_bias) + (1 - self.beta1) * bias_gradient

            # Update second moment estimate
            v_weights, v_bias = self.v[i]

            v_weights = (self.beta2 * v_weights) + (1 - self.beta2) * (weights_gradient ** 2)
            v_bias = (self.beta2 * v_bias) + (1 - self.beta2) * (bias_gradient ** 2)

            # Store updated parameters
            self.m[i] = (m_weights, m_bias)
            self.v[i] = (v_weights, v_bias)

            # correction
            m_weights_hat = m_weights / (1 - self.beta1 ** self.t)
            m_bias_hat = m_bias / (1 - self.beta1 ** self.t)
                        
            v_weights_hat = v_weights / (1 - self.beta2 ** self.t)
            v_bias_hat = v_bias / (1 - self.beta2 ** self.t)
            
            # Update parameters
            layer.weights = layer.weights - learning_rate * (m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon))
            layer.bias = layer.bias - learning_rate * (m_bias_hat / (np.sqrt(v_bias_hat) + self.epsilon))
