'''
Run test with `pytest`
'''

from collections import Counter
import numpy as np
import pytest
import random

from .balanced_batch_sampler import BalancedBatchSampler

TEST_DATASET_SIZE=1000

class TestDataset:
    def __init__(self, num_samples=1000, imbalance_ratio=7, seed=1):
        self.rand = random.Random(seed)
        self.imbalance_ratio = imbalance_ratio
        self.data = []
        for i in range(num_samples):
            if i % (self.imbalance_ratio + 1) == 0:
                label = 1
            else:
                label = 0
            row = (i, None, None, label, None)
            self.data.append(row)
        self.rand.shuffle(self.data)

    def __iter__(self):
        return (d for d in self.data)

    def __len__(self):
        return len(self.data)

@pytest.fixture
def test_dataset():
    dataset = TestDataset(num_samples=TEST_DATASET_SIZE)
    return dataset

def count_labels_in_batch(batch, dataset):
    labels = []
    for row_idx in batch:
        (_, _, _, label, _) = dataset.data[row_idx]
        labels.append(label)
    return Counter(labels)

def test_construct_balanced_batch_sampler(test_dataset):
    batch_size=8
    batch_sampler = BalancedBatchSampler(test_dataset, batch_size=batch_size, drop_last=False)

def test_batch_size_too_small(test_dataset):
    '''
    If the batch size supplied to the batch sampler is too small, then we can't guarantee that each batch
    will contain both labels. Test that an error is thrown in this case.
    '''
    batch_size=7
    with pytest.raises(ValueError):
        BalancedBatchSampler(test_dataset, batch_size=batch_size, drop_last=False)

def test_balanced_batch_sampler_perfect_batches(test_dataset):
    '''
    Test that if we have a label imbalance ratio of 7:1, and we sample batches of 8 samples, each
    batch will contain exactly 7 negatives and 1 positive.
    '''
    batch_size=8
    batch_sampler = BalancedBatchSampler(test_dataset, batch_size=batch_size, drop_last=False)

    num_batches = TEST_DATASET_SIZE/batch_size
    batches = []
    i = 0
    for batch in batch_sampler:
        i += 1
        batch_label_counts = count_labels_in_batch(batch, test_dataset)
        assert batch_label_counts[0] == 7
        assert batch_label_counts[1] == 1
        batches.append(batch)

    # In this situation, the dataset perfectly divides into the number of batches. Verify this.
    assert len(batches) == TEST_DATASET_SIZE/batch_size

def test_balanced_batch_sampler_imperfect_batches(test_dataset, tolerance=0.1):
    '''
    Test that even if we can't perfectly distribute minority class points across batches, the mean
    absolute error of the imbalance ratio in each batch (compared to the entire dataset) is small.
    '''
    batch_size=15
    batch_sampler = BalancedBatchSampler(test_dataset, batch_size=batch_size, drop_last=False)
    batches = []

    true_minority_ratio = 1.0 / (1 + test_dataset.imbalance_ratio)
    absolute_minority_ratio_errors = []
    for batch in batch_sampler:
        batches.append(batch)
        batch_label_counts = count_labels_in_batch(batch, test_dataset)
        minority_imbalance_ratio = float(batch_label_counts[1]) / (batch_label_counts[0] + batch_label_counts[1])
        # Each batch should still contain samples from both classes
        assert len(batch_label_counts) == 2
        absolute_minority_ratio_error = np.abs(minority_imbalance_ratio - true_minority_ratio)
        absolute_minority_ratio_errors.append(absolute_minority_ratio_error)

    # The mean average error in the proportion of each batch consisting of negative examples should be close
    # to the full-dataset proportion of negative examples.
    assert np.mean(absolute_minority_ratio_errors) < tolerance
    assert len(batches) == np.ceil(float(TEST_DATASET_SIZE)/batch_size)