from typing import Dict, Callable
import numpy as np

#layer interface
class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, np.ndarray] = {} #constants in eqs
        self.gradients: Dict[str, np.ndarray] = {}

    def forward(self, inputs: np.ndarray):
        raise NotImplementedError
    
    def backward(self, inputs: np.ndarray):
        raise NotImplementedError
    


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        #init with random parmas initally
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return

class ActivationFunc(Layer):
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        self.func = function

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.func(input)
    
    #TODO: define activation functions