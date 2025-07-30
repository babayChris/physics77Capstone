"""
Optimizers depend on the model that you are going with:
    Assume that our dataset has been loaded, normalized, split, the model has been defined, 
    and then we have a defined function as well. This is just a model for the simplest kind of
    machine learning--linear regression--and we can always update as necessary.
"""
import numpy as np
from network import SequentialNetwork
from layer import Linear

class Optimizer():
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def apply_gradients(self, model: SequentialNetwork, learning_rate: float):
        for layer in model.layers:
            if hasattr(layer, 'grad') and 'w' in layer.grad and 'b' in layer.grad:
                layer.params['w'] -= learning_rate * layer.grad['w']
                layer.params['b'] -= learning_rate * layer.grad['b']