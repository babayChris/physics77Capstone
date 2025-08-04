import numpy as np


class Loss:
    """
    loss takes in predicted (forward pass return value) and ground truth values and runs a loss function

    """

    def __init__(self):
        pass

    def loss(self, pred: np.ndarray, truth: np.ndarray) -> float:
        if truth.ndim > 1:
            truth = truth.flatten()
        
        pred_clipped = np.clip(pred, 1e-15, 1 - 1e-15)
        
        probs = pred_clipped[np.arange(len(truth)), truth.astype(int)]
        
        loss = -np.mean(np.log(probs))
        
        return loss

        
    def gradLoss(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        calculates gradient of loss for sparse categorical cross entropy 
        """
        if truth.ndim > 1:
            truth = truth.flatten()
            
        truth_one_hot = np.zeros_like(pred)
        truth_one_hot[np.arange(len(truth)), truth.astype(int)] = 1
        
        return (pred - truth_one_hot) / pred.shape[0]
