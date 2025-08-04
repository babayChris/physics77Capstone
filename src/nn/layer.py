from typing import Dict, Callable
import numpy as np

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, np.ndarray] = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # He-Normal distributition for initialization
        self.params['w'] = np.random.normal(0, np.sqrt(2 / input_size), (input_size, output_size))
        self.params['b'] = np.zeros(output_size)
        
        self.grad: Dict[str, np.ndarray] = {
            'w': np.zeros_like(self.params['w']), 
            'b': np.zeros_like(self.params['b'])
        }

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:        
        self.grad['w'] = stored_input.T @ gradient
        self.grad['b'] = np.sum(gradient, axis=0)
        dx = gradient @ self.params['w'].T
        return dx


class ActivationFunc(Layer):
    def __init__(self, function_name: str):
        super().__init__()
        self.function_name = function_name
        self.stored_input = None
        
        self.funcDict = {
            "relu": self.relu,
            "softmax": self.softmax,
        }
        
        self.gradDict = {
            "relu": self.gradRelu,
            "softmax": self.gradSoftmax
        }

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.stored_input = input
        return self.funcDict[self.function_name](input)

    def backward(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:
        return self.gradDict[self.function_name](gradient, self.stored_input)

    def relu(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def softmax(self, inputs: np.ndarray) -> np.ndarray:
        shifted_inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def gradRelu(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:        
        return gradient * (stored_input > 0).astype(float)

    def gradSoftmax(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:
        return gradient
        



