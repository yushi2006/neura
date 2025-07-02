import numpy as np

from ..core.tensor import Tensor


class DataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data_list = [self.dataset[j][:-1] for j in batch_indices]
            batch_labels_list = [self.dataset[j][-1] for j in batch_indices]

        batch_data_np = np.array(batch_data_list)
        batch_labels_np = np.array(batch_labels_list)

        data_tensor = Tensor(batch_data_np, requires_grad=False)
        labels_tensor = Tensor(batch_labels_np, requires_grad=False)

        yield data_tensor, labels_tensor
