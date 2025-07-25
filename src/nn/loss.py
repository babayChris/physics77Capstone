import numpy as np


class Loss:
    """
    loss takes in predicted (forward pass return value) and ground truth values and runs a loss function

    """
    def loss(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """
        computes total loss for one forward pass
        """
        raise NotImplementedError
        
    def gradLoss(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        calculates gradient tensor used in backward pass
        """
        raise NotImplementedError
    
#TODO: impliment common Loss functions to test in network