import numpy as np

from typing import NamedTuple, Iterator

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
        numSampes = len(inputs)
        batch_start_index = np.arange(0, len(inputs), self.batch_size)
        
        if self.shuffle:
            np.random.shuffle(batch_start_index)

        for batch in batch_start_index:
            batch_end_index = batch + self.batch_size
            input_batch = inputs[batch: batch_end_index]
            test_batch = truth[batch: batch_end_index]
            yield Batch(input_batch, test_batch)
