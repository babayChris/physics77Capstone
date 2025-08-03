import numpy as np
import pandas as pd
from typing import NamedTuple, Iterator, List

"""

input: [Temperature (K), Luminosity(L/Lo), Radius(R/Ro), Absolute magnitude(Mv)]


"""

Batch = NamedTuple("Batch", [("inputs", np.ndarray),("truth", np.ndarray)])


class BatchIterator():
    def __init__(self, batch_size: int, shuffle: bool):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: np.ndarray, truth: np.ndarray) -> Iterator[Batch]:
        """
        executes when BatchIterator() is called w/ some args defined in func
        ndarray inputs - input tensors
        ndarray truth - ground truth tensors
        batch_arr stores starting index for each batch in input ndarray
        for end index do starting index + batch_size
        """
        # numSampes = len(inputs)
        batch_start_index = np.arange(0, len(inputs), self.batch_size)
        
        if self.shuffle:
            np.random.shuffle(batch_start_index)

        for batch in batch_start_index:
            batch_end_index = batch + self.batch_size
            input_batch = inputs[batch: batch_end_index]
            test_batch = truth[batch: batch_end_index]
            yield Batch(input_batch, test_batch)


DataTuple = NamedTuple("DataTuple", [("train", BatchIterator), ("test", BatchIterator)])

class Data():
    def __init__(self, dataset: pd.DataFrame, 
                 truth_col: str,
                 test_ratio: int, 
                 batch_size: int, 
                 dropArr: List[str], 
                 shuffle: bool = True,):
        #preprocess pd dataframe
        #TODO: check if truth_col exists
        truth = np.array(dataset[truth_col])
        dataset = np.array(dataset.drop(dropArr, axis=1))
        truth = truth.reshape(-1,1)

        #split
        split_index = int(len(dataset) * test_ratio)
        self.test_inputs = dataset[:split_index,:]
        self.test_truths = truth[:split_index,:]
        self.train_inputs = dataset[split_index:,:]
        self.train_truths = truth[split_index:,:]
        #NOTE: no point in shuffling the test batches
        self.data = DataTuple(BatchIterator(batch_size, shuffle), BatchIterator(batch_size, False)) 

    def get_train_iter(self) -> Iterator[Batch]:
        return self.data.train(self.train_inputs, self.train_truths)
    
    def get_test_iter(self) -> Iterator[Batch]:
        return self.data.test(self.test_inputs, self.test_truths)