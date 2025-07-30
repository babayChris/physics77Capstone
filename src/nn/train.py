from layer import Linear
from data import BatchIterator, DataIterator
from network import Network
import numpy as np


def train(numEpochs: int,
          inputs: np.ndarray,
          truth: np.ndarray,
          iterator: DataIterator = BatchIterator(),
          net: Network = Network()
          ):
    """
    int numEpochs - number of training passes

    for each epoch
        1) reset loss to 0
        per batch
            1) forward pass (make prediction)
            2) calc loss for logging purposes
            3) calc gradients
            4) backward pass with grads
            5) run optimizer on neural net
    """
    for epoch in range(numEpochs):
        currLoss = 0.0 
        for batch in BatchIterator(inputs, truth):
                prediction = net.
                