from typing import List, Dict
from layer import Layer, Linear, ActivationFunc
import numpy as np

class SequentialNetwork():
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.layer_inputs: List[np.array] = []

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layer_inputs = []
        current_input = inputs
        
        for layer in self.layers:
            self.layer_inputs.append(current_input)
            current_input = layer.forward(current_input)
        
        return current_input

    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        self.gradient_reset()
        
        current_gradient = loss_gradient
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_input = self.layer_inputs[i]
            current_gradient = layer.backward(current_gradient, layer_input)
        
        return current_gradient

    def gradient_reset(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                if hasattr(layer, 'grad') and layer.grad:
                    layer.grad['w'] = np.zeros_like(layer.params['w'])
                    layer.grad['b'] = np.zeros_like(layer.params['b'])


