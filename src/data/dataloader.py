import numpy as np

from ..core.tensor import Tensor


class DataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not isinstance(dataset, np.ndarray):
            print("Warning: For best performance, dataset should be a NumPy array.")

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            batch_data = self.dataset[batch_indices, :-1]
            batch_labels = self.dataset[batch_indices, -1]

            if batch_labels.ndim == 1:
                batch_labels = batch_labels.reshape(-1, 1)

            yield (
                Tensor(batch_data, requires_grad=False),
                Tensor(batch_labels, requires_grad=False),
            )
