"""hub to set up and run everything"""
from typing import List, Dict, Union
import numpy as np

from network import SequentialNetwork
from layer import Layer, Linear, ActivationFunc
from loss import Loss
from optimizer import Optimizer, adamOptimizer
from data import BatchIterator

class Model():
    def __init__(self):
        self.network = SequentialNetwork([])
        self.metrics: Dict = {'train_loss': []}
        self.loss_funct = Loss()

    def addLayer(self, layer: Layer):
        self.network.layers.append(layer)
        
    def compile(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.optimizer = adamOptimizer(learning_rate)
  
    def train(self, inputs: np.ndarray, truth: np.ndarray, epochs: int, batch_size: int, shuffle: bool = True):
        iterator = BatchIterator(batch_size=batch_size, shuffle=shuffle)
        print('Starting training:')

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in iterator(inputs, truth):
                predictions = self.network.forward(batch.inputs)
                
                batch_loss = self.loss_funct.loss(predictions, batch.truth)
                
                loss_grad = self.loss_funct.gradLoss(predictions, batch.truth)
                
                self.network.backward(loss_grad)
                
                self.optimizer.apply_gradients_adam(self.network)
                
                epoch_loss += batch_loss
                num_batches += 1
                
            avg_loss = epoch_loss / num_batches
            self.metrics['train_loss'].append(avg_loss)
            
            print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}')

        print('Training completed!')
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.network.forward(inputs)
        
        