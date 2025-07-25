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
        """
        weight * input + bias
        y = mx + b
        """
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        #TODO
        return

class ActivationFunc(Layer):
<<<<<<< HEAD
    def __init__(self):
        self.funcDict = {
            "sigmoid" : self.sigmoid,
            "relu" : self.relu,
            "softmax" : self.softmax,
            "leaky_relu" : self.leaky_relu,
            "tanh" : self.tanh
        }
=======
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        self.func = function
>>>>>>> 0b02b581a4f59bf58aa6cff7df259d0fb5d5ba7b

    def forward(self, input: np.ndarray, func: str) -> np.ndarray:
        if func not in self.funcDict:
            return KeyError
        return self.funcDict[func](input)

    def sigmoid(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(input))

    def relu(self, input: np.ndarray) -> np.ndarray:
        return np.max(0, input)

    def softmax(self, input: np.ndarray) -> np.ndarray:
        e_x = np.exp(input - np.max(input))
        return e_x / sum(e_x, axis = 0)

    def leaky_relu(self, input: np.ndarray, alpha = 0.01) -> np.ndarray: #only works for alpha<1
        return np.max(alpha * input, input)

    def tanh(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)
    #TODO: define activation functions
    #update: Sigmoid, ReLu, softmax, leaky ReLu, tanh functions added
