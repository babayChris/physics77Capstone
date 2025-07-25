import numpy as np


class Loss:
    """
    loss takes in predicted (forward pass return value) and ground truth values and runs a loss function

    """

    def __init__(self):
        pass

    def loss(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """
        computes aggregated loss for one forward pass using sparse catigorical entropy loss
        """

        predClipped = np.clip(pred, a_min = 10**(-15),a_max = 1 )
        truthOneHot = np.zeros_like(pred)
        truthOneHot[np.arange(len(truth)), truth.astype(int)] = 1

        loss = -truthOneHot * np.log(predClipped)

        totalLoss = np.sum(loss,axis = -1)   

        aggregatedLoss = np.mean(totalLoss)
        
        return aggregatedLoss
        
    def gradLoss(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        calculates gradient tensor used in backward pass
        """
        raise NotImplementedError
    
#TODO: impliment common Loss functions to test in network
