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

        logSoft = self.logSoftmax(pred) 
    
        probs = logSoft[np.arange(len(truth)), truth.astype(int)]
        loss = -np.mean(probs)
        
        return loss

        
    def gradLoss(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        calculates gradient of loss for sparse categorical cross entropy 
        """
        sofmaxPred = np.exp(self.logSoftmax(pred))
        if truth.ndim > 1:
            truth = truth.flatten()
            
        truth_one_hot = np.zeros_like(pred)
        truth_one_hot[np.arange(len(truth)), truth.astype(int)] = 1
        
        return (sofmaxPred - truth_one_hot) / pred.shape[0]


    def logSoftmax(self, pred: np.ndarray) -> np.ndarray:
        shifted_inputs = pred - np.max(pred, axis=-1, keepdims=True)
        return shifted_inputs - np.log(np.sum(np.exp(shifted_inputs), axis=1, keepdims=True))
        truth_one_hot[np.arange(len(truth)), truth.astype(int)] = 1
        
        return (pred - truth_one_hot) / pred.shape[0]
