import unittest

import numpy as np

# --- Mocks for Self-Contained Testing ---
from neura.data import DataLoader


# A simple mock Tensor class that mimics your real one for testing purposes.
class MockTensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.shape = self.data.shape

    def __repr__(self):
        return f"MockTensor({self.data})"


# We will replace our mock with your actual Tensor class for the real test.
# from my_framework.tensor import Tensor as RealTensor
# In this file, we just alias the mock to Tensor.
Tensor = MockTensor

# --- The Test Suite ---


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock dataset that will be used across all tests.
        The dataset has 10 items. Each data item is a unique 2x2 array.
        """
        self.dataset = []
        for i in range(10):
            # Each data item is a numpy array filled with its index 'i'
            # This makes it easy to verify data correctness later.
            data = np.full((2, 2), i, dtype=np.float32)
            label = i
            self.dataset.append((data, label))

    def test_initialization(self):
        """Test if the DataLoader initializes with correct attributes."""
        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        self.assertEqual(loader.batch_size, 4)
        self.assertFalse(loader.shuffle)
        self.assertIs(loader.dataset, self.dataset)

    def test_len_calculation(self):
        """Test if the __len__ method calculates the number of batches correctly."""
        # Case 1: Perfect division
        loader_even = DataLoader(self.dataset, batch_size=5)
        self.assertEqual(len(loader_even), 2)  # 10 / 5 = 2 batches

        # Case 2: Imperfect division (edge case)
        loader_uneven = DataLoader(self.dataset, batch_size=3)
        self.assertEqual(
            len(loader_uneven), 4
        )  # 10 / 3 = 3 full batches + 1 smaller batch

        # Case 3: Batch size larger than dataset
        loader_large_batch = DataLoader(self.dataset, batch_size=12)
        self.assertEqual(len(loader_large_batch), 1)

    def test_iteration_and_batch_size(self):
        """Test iteration and verify that batch sizes are correct, including the last one."""
        batch_size = 3
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        batches = list(iter(loader))

        # Check that the number of yielded batches is correct
        self.assertEqual(len(batches), 4)

        # Check the shapes of the data tensors in each batch
        # Expected batch sizes: 3, 3, 3, 1
        self.assertEqual(batches[0][0].shape, (3, 1, 2, 2))  # data tensor
        self.assertEqual(batches[1][0].shape, (3, 1, 2, 2))
        self.assertEqual(batches[2][0].shape, (3, 1, 2, 2))
        self.assertEqual(batches[3][0].shape, (1, 1, 2, 2))  # last, smaller batch

        # Check the shapes of the label tensors
        self.assertEqual(batches[0][1].shape, (3,))  # label tensor
        self.assertEqual(batches[3][1].shape, (1,))

    def test_data_correctness_no_shuffle(self):
        """Verify that the data and labels yielded are correct and in order when shuffle=False."""
        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)

        # Unpack all data and labels from the loader
        all_data = []
        all_labels = []
        for data_batch, label_batch in loader:
            self.assertIsInstance(data_batch, Tensor)
            self.assertIsInstance(label_batch, Tensor)
            # Squeeze to remove the channel dimension we added
            all_data.extend(list(np.squeeze(data_batch.data, axis=1)))
            all_labels.extend(list(label_batch.data))

        # Verify that the reassembled data matches the original dataset
        for i in range(10):
            original_data, original_label = self.dataset[i]
            # Check labels
            self.assertEqual(all_labels[i], original_label)
            # Check data content
            np.testing.assert_array_equal(all_data[i], original_data)

    def test_shuffling(self):
        """Verify that shuffle=True changes the order of the data."""
        loader = DataLoader(self.dataset, batch_size=10, shuffle=True)

        # Get the order of items from two separate iterations
        run1_labels = list(next(iter(loader))[1].data)
        run2_labels = list(next(iter(loader))[1].data)

        # The probability of two shuffles being identical for 10 items is 1/10! (very low)
        # This is a standard way to test shuffling.
        self.assertNotEqual(
            run1_labels, list(range(10)), "Data was not shuffled from original order."
        )
        self.assertNotEqual(
            run1_labels, run2_labels, "Two shuffles produced the exact same order."
        )

        # A more robust check: ensure that all original data is still present, just reordered.
        self.assertCountEqual(
            run1_labels, list(range(10)), "Shuffling lost or duplicated data."
        )


# To run the tests, you would have your DataLoader class defined here or imported
# For this example, I'll paste the class we designed earlier.


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data_list = [np.array(self.dataset[j][0]) for j in batch_indices]
            batch_labels_list = [self.dataset[j][1] for j in batch_indices]
            batch_data_np = np.array(batch_data_list)
            batch_labels_np = np.array(batch_labels_list)
            if batch_data_np.ndim == 3:
                batch_data_np = np.expand_dims(batch_data_np, axis=1)
            yield Tensor(batch_data_np), Tensor(batch_labels_np)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    unittest.main()
