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
        # Normal distributition for initialization (switched from He-normal to normal per instructer recomendation)
        self.params['w'] = np.random.normal(0, 1, (input_size, output_size))
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
        }
        
        self.gradDict = {
            "relu": self.gradRelu,
        }

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.stored_input = input
        return self.funcDict[self.function_name](input)

    def backward(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:
        return self.gradDict[self.function_name](gradient, self.stored_input)

    def relu(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def gradRelu(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:        
        return gradient * (stored_input > 0).astype(float)




