from typing import List
from layer import Layer
import numpy as np

class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    
class SequentialNetwork():
    
