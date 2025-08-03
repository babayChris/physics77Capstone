from typing import List
from model import Model
from layer import Layer, Linear, ActivationFunc
from pandas import DataFrame

class ModelBuilder():
    def __init__(self):
        self.model = Model()

    def hidden_layers(self, layers_list: List[int], 
                      layer_type: Layer = None, activation: str = 'relu'):
        for i in range(1, len(layers_list)):
            if layer_type is None:
                self.model.addLayer(Linear(layers_list[i-1], layers_list[i]))
                self.model.addLayer(ActivationFunc(activation))
            else:
                self.model.addLayer(layer_type(layers_list[i-1], layers_list[i]))
                self.model.addLayer(ActivationFunc(activation))


    def build(self, layers_list: List[int]):
        #reset model
        self.model = Model()
        #model will always start with 4 input nodes
        self.model.addLayer(Linear(4, layers_list[0]))
        self.model.addLayer(ActivationFunc("relu"))
        if len(layers_list) > 1:
            self.hidden_layers(layers_list)
        self.model.addLayer(Linear(layers_list[-1], 6))
        self.model.addLayer(ActivationFunc("softmax"))
    
    def save_model(self, path: str = "load/model_info.npz"):
        self.model.save_model(path)
    
    def load_model(self, path: str = "load/model_info.npz"):
        self.model.load_model_from_path(path)
    
    def compile(self, data: DataFrame):
        self.model.compile(data)

    def train(self, data: DataFrame, epoch: int = 100):
        self.compile(data)
        self.model.train(epoch)