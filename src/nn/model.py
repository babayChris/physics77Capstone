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
        self.metrics: Dict = {'train_loss': [], 'train_accuracy' : [], 'val_pred' : [], 'val_pred_loss' : [], 'val_pred_accuracy' : []}
        self.loss_funct = Loss()
        self.num_epochs = 0

    def addLayer(self, layer: Layer):
        self.network.layers.append(layer)
        
    def compile(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.optimizer = adamOptimizer(learning_rate)
  
    def train(self, inputs: np.ndarray, truth: np.ndarray, epochs: int, batch_size: int, val_inputs: np.ndarray, val_truth: np.ndarray, shuffle: bool = True):
        iterator = BatchIterator(batch_size=batch_size, shuffle=shuffle)
        print('Starting training:')

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accu = 0
            num_batches = 0

            for batch in iterator(inputs, truth):
                predictions = self.network.forward(batch.inputs)
                
                batch_loss = self.loss_funct.loss(predictions, batch.truth)

                batch_accu = self.accu(predictions, batch.truth)
                
                loss_grad = self.loss_funct.gradLoss(predictions, batch.truth)
                
                self.network.backward(loss_grad)
                
                self.optimizer.apply_gradients_adam(self.network)
                
                epoch_loss += batch_loss
                epoch_accu += batch_accu
                num_batches += 1

            test_pred = self.predict(val_inputs)
            val_loss = self.loss_funct.loss(test_pred, val_truth)
            
            
            
            avg_loss = epoch_loss / num_batches
            avg_accu = epoch_accu / num_batches
            val_accu = self.accu(test_pred, val_truth)
            
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['val_pred'].append(test_pred)
            self.metrics['val_pred_loss'].append(val_loss)
            self.metrics['train_accuracy'].append(avg_accu)
            self.metrics['val_pred_accuracy'].append(val_accu)
            
            print(f'Epoch: {self.num_epochs + epoch + 1}, Loss: {avg_loss:.4f}')

        print('Training completed!')
        self.num_epochs += epochs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        result = self.network.forward(inputs)
        results = np.exp(self.loss_funct.logSoftmax(result))
        return results

    def accu(self, pred: np.ndarray, truth: np.ndarray) -> float:
        numCorrect = 0
        preds = np.argmax(pred, axis=1)
        for i in range(len(truth)):
            if preds[i] == truth[i]:
                numCorrect +=1

        accuracy = numCorrect/len(truth)
        return accuracy

        
        