"""hub to set up and run everything"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import os

from network import SequentialNetwork
from layer import Layer, Linear, ActivationFunc
from loss import Loss
from optimizer import Optimizer
from data_load import Data

class Model():
    def __init__(self):
        self.network = SequentialNetwork([])
        self.metrics: Dict([str, List[float]]) = {'train loss': []}
        self.loss_funct = Loss()

    def addLayer(self, layer: Layer):
        self.network.layers.append(layer)
        
    def compile(self, dataset: pd.DataFrame, 
                learning_rate: float = 0.000001, 
                batch_size: int = 8, shuffle: bool = True, 
                test_ratio: int = 0.15,
                drop_array: List[str] = ['Star type', 'Star color', 'Spectral Class'],
                truth_col: str  = 'Star type'
                ):
        self.learning_rate = learning_rate
        self.optimizer = Optimizer(learning_rate)
        self.data: Data = Data(dataset, truth_col, test_ratio, batch_size, drop_array, shuffle)

    def train(self, epochs: int):
        print('starting training:')

        for epoch in range(epochs):
            curr_loss = 0
            num_batches = 0

            iterator = self.data.get_train_iter()
            for batch in iterator:
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
    
    #Test func -> run weights
    def test(self) -> Tuple:
        """
        want to return error with params 
        (List[nodes](index is number layer - 1), error)
        TODO
        """
        test_iter = self.data.get_test_iter()
        tot_loss = 0
        tot_batches = 0

        for batch in test_iter:
            pred = self.network.forward(batch.inputs)
            batch_loss = self.loss_funct.loss(pred, batch.truth)
            
            tot_loss += batch_loss
            tot_batches += 1

        avg_test_loss = tot_loss / tot_batches

        return avg_test_loss

    def save_model(self, filepath: str):
        print('-------' \
              'SAVING'\
              '-------')
        
        save_dict = {}
        layer_count = 0
        arch_info = []
        for i, layer in enumerate(self.network.layers):
            if isinstance(layer, Linear):
                save_dict[f'weight_{layer_count}'] = layer.params['w']
                save_dict[f'bias_{layer_count}'] = layer.params['b']

                arch_info.append({
                    'layer_type': 'Linear', #NOTE: hardcoded if we do not experiment
                    'input_size': layer.params['w'].shape[0],
                    'output_size': layer.params['w'].shape[1],
                    'nodes': layer.nodes
                })
                layer_count += 1

            elif isinstance(layer, ActivationFunc):
                arch_info.append({
                'layer_type': 'ActivationFunc',
                'function_name': layer.function_name
            })
        save_dict['architecture_info'] = np.array(arch_info, dtype=object)

        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    def load_model_from_path(self, filepath: str = "load/model_info.npz"):
        print('-------'\
              'LOADING'\
                '-------')
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        if not os.path.exists(filepath):
            raise FileNotFoundError("Invalid load file")
        
        data = np.load(filepath, allow_pickle=True)

        arch_info = data['architecture_info']

        layer_index = 0
        network = []
        for layer in arch_info:
            if layer['layer_type'] == 'Linear':
                input_size = int(layer['input_size'])
                output_size = int(layer['output_size'])

                linear_layer = Linear(input_size, output_size)

                linear_layer.params['w'] = data[f'weight_{layer_index}']
                linear_layer.params['b'] = data[f'bias_{layer_index}']

                network.append(linear_layer)

                layer_index += 1
            elif layer['layer_type'] == 'ActivationFunc':
                activation = ActivationFunc(layer['function_name'])
                network.append(activation)

        self.network = network

        print('loaded model!')

