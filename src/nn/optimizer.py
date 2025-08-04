"""
Optimizers depend on the model that you are going with:
    Assume that our dataset has been loaded, normalized, split, the model has been defined, 
    and then we have a defined function as well. This is just a model for the simplest kind of
    machine learning--linear regression--and we can always update as necessary.
"""
import numpy as np
from network import SequentialNetwork
from layer import Linear, ActivationFunc

class Optimizer():
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def apply_gradients(self, model: SequentialNetwork, learning_rate: float):
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.params['w'] -= learning_rate * layer.grad['w']
                layer.params['b'] -= learning_rate * layer.grad['b']

class adamOptimizer():
    def __init__(self, learning_rate: float, beta_1: float=0.9, beta_2: float=0.999, epsi: float=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsi = epsi
        self.t = 0
        self.m = {}
        self.v = {}

    def apply_gradients_adam(self, model: SequentialNetwork):
        self.t += 1

        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                if i not in self.m:
                    self.m[i] = {'w': np.zeros_like(layer.params['w']), 'b': np.zeros_like(layer.params['b'])}
                    self.v[i] = {'w': np.zeros_like(layer.params['w']), 'b': np.zeros_like(layer.params['b'])}

                grad_w = layer.grad['w']
                grad_b = layer.grad['b']

                self.m[i]['w'] = self.beta_1 * self.m[i]['w'] + (1 - self.beta_1) * grad_w
                self.m[i]['b'] = self.beta_1 * self.m[i]['b'] + (1 - self.beta_1) * grad_b

                self.v[i]['w'] = self.beta_2 * self.v[i]['w'] + (1 - self.beta_2) * (grad_w ** 2)
                self.v[i]['b'] = self.beta_2 * self.v[i]['b'] + (1 - self.beta_2) * (grad_b ** 2)

                m_w = self.m[i]['w'] / (1.0 - self.beta_1**self.t)
                v_w = self.v[i]['w'] / (1.0 - self.beta_2**self.t)
                m_b = self.m[i]['b'] / (1.0 - self.beta_1**self.t)
                v_b = self.v[i]['b'] / (1.0 - self.beta_2**self.t)

                layer.params['w'] -= self.learning_rate * m_w / (np.sqrt(v_w) + self.epsi)
                layer.params['b'] -= self.learning_rate * m_b / (np.sqrt(v_b) + self.epsi)
