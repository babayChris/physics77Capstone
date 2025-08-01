"""hub to set up and run everything"""
from typing import List, Dict, Union
import numpy as np

from network import SequentialNetwork
from layer import Layer, Linear, ActivationFunc
from loss import Loss
from optimizer import Optimizer
from data import BatchIterator

class Model():
    def __init__(self):
        self.network = SequentialNetwork([])
        self.metrics: Dict([str, List[float]]) = {'train loss': []}
        self.loss_funct = Loss()

    def addLayer(self, layer: Layer):
        self.network.layers.append(layer)
        
    def compile(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.optimizer = Optimizer(learning_rate)
  
    def train(self, inputs: np.ndarray, truth: np.ndarray, epochs: int, batch_size: int, shuffle: bool = True):
        
        iterator = BatchIterator(batch_size = batch_size, shuffle=shuffle)

        print('starting training:')

        for epoch in range(epochs):
            curr_loss = 0
            num_batches = 0

            for batch in iterator(inputs, truth):
                predictions = self.network.forward(batch.inputs)
                truth_one_hot = np.zeros_like(predictions)
                
                if batch.truth.ndim == 1:
                    truth_indices = batch.truth
                elif batch.truth.ndim == 2 and batch.truth.shape[1] == 1:
                    truth_indices = batch.truth.flatten()
                else:
                    truth_one_hot = batch.truth 

                batch_loss = self.loss_funct.loss(predictions, batch.truth)

                loss_grad = self.loss_funct.gradLoss(predictions, batch.truth)
                curr_loss += batch_loss
                num_batches += 1

                self.network.backward(loss_grad)
                self.optimizer.apply_gradients(self.network,self.optimizer.learning_rate)
                
            self.metrics['train loss'].append(curr_loss / num_batches)

            print(f'epoch {epoch + 1} complete, with average loss of {self.metrics["train loss"][-1]:0.4f}')

        print('All Done Training!')

    def useTrainedModel(self, inputs: np.ndarray) -> np.ndarray:
        return self.network.forward(inputs)
        