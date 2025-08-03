from typing import Dict, Callable
import numpy as np

#layer interface
class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, np.ndarray] = {} #constants in eqs
        #self.gradients: Dict[str, np.ndarray] = {}

    
class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.nodes = output_size
        #init with random parmas initally
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)
        self.grad: Dict[str, np.ndarray] = {
            'w': np.zeros_like(self.params['w']), 
            'b': np.zeros_like(self.params['b'])
        }

    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        weight * input + bias
        y = mx + b
        """
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, gradient: np.ndarray,backward_inputs: np.ndarray) -> np.ndarray:
        self.grad['w'] = backward_inputs.T @ gradient #dL/dw
        self.grad['b'] = np.sum(gradient, axis = 0) #dL/db
        dx = gradient @ self.params['w'].T #dL/dx
        return dx

    def get_nodes(self):
        return self.nodes
    
    def load_weights(self, weights: np.ndarray, bias: np.ndarray):
        self.params['w'] = weights
        self.params['b'] = bias

        
class ActivationFunc(Layer):
    def __init__(self, function_name: str):
        super().__init__()
        self.function_name = function_name
        self.stored_input = None
        
        self.funcDict = {
            "relu" : self.relu,
            "softmax" : self.softmax,
        }

        self.gradDict = {
            "relu" : self.gradRelu,
            "softmax" : self.gradSoftMax
        }
        

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.stored_input = input
        return self.funcDict[self.function_name](input)

    def backward(self, gradient: np.ndarray, inputs_for_backwards: np.ndarray) -> np.ndarray:
        return self.gradDict[self.function_name](gradient, self.stored_input)

    def relu(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def softmax(self, inputs: np.ndarray) -> np.ndarray:
        e_x = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))        
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradRelu(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:
        gradRelu = (stored_input > 0).astype(float)
        return gradient*gradRelu


    def gradSoftMax(self, gradient: np.ndarray, stored_input: np.ndarray) -> np.ndarray:
        return gradient

