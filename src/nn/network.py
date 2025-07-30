from typing import List, Dict
from layer import Layer
import numpy as np

class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        goes through layers backwards 
        """
        for layer in reversed(self.layers): 
            grad = layer.backward(grad)
        return grad