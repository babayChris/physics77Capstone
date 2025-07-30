from typing import List, Dict
from layer import Layer, Linear, ActivationFunc
import numpy as np

class SequentialNetwork():
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.layer_inputs: List[np.array()] = []

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layer_inputs = []
        for layer in self.layers:
            self.layer_inputs.append(inputs)
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grady: np.ndarray) -> np.ndarray:
        
        self.gradient_reset()

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            was_input = self.layer_inputs[i]
            grady = layer.backward(grady, was_input)
            
        return grady
        
    def gradient_reset(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                if hasattr(layer, 'grad') and layer.grad:
                    if 'w' in layer.params:
                        layer.grad['w'] = np.zeros_like(layer.params['w'])
                    if 'b' in layer.params:
                        layer.grad['b'] = np.zeros_like(layer.params['b'])


