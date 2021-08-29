import numpy as np
import random
from torch.utils.data import Dataset, RandomSampler, Sampler
from typing import Iterator, List

class SingleLabelSampler(Sampler):
    r"""Torch Sampler that samples points from a dataset with a given label, to construct balanced batches.

    Args:
        dataset (Dataset): Drug Synergy Dataset to sample from
        label_to_keep (int): Only sample points whose label index matches this value
        shuffle (bool): Whether or not to construct batches from shuffled datapoints
        rand (random.Random): Random number generator to use for shuffling
    """
    def __init__(self, dataset: Dataset, label_to_keep: int, shuffle: bool = False, rand: random.Random = None):
        self.matching_indices = [i for (i, (_, _, _, label, _)) in enumerate(dataset) if label == label_to_keep]
        if shuffle:
            rand.shuffle(self.matching_indices)

    def __iter__(self):
        # Convert the list of indices to a generator.
        return (i for i in self.matching_indices)

    def __len__(self):
        return len(self.matching_indices)


class BalancedBatchSampler:
    r"""Class to yield a mini-batch of indices. We have two desired attributes for each batch:
    1) Each batch contains both positives and negative examples
    2) Each batch otherwise has a similar label distribution to the entire dataset

    Args:
        dataset (Dataset): Dataset to construct batches for
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        shuffle (bool): Whether or not to construct batches from shuffled datapoints
        seed (int): Random seed to use for shuffling

    Algorithm Explanation:
    We have one label that occurs more frequently than the other; we call these the "majority class" and "minority class".
    In order to capture our 2 goals here, we force every batch to contain a sample from the minority class, and then sample 
    the remaining samples with P(minority class) = (batch size / (imbalance ratio + 1) - 1) / (batch size - 1).

    where "imbalance ratio" := (# of points with the majority label) / (# of points with the minority label).

    We derive this probability as follows:
    1) For any point in the dataset:  P(minority label) = 1 / (imbalance ratio + 1)
    2) Expected_Value[number of minority samples in a batch] = batch size * 1 / (imbalance ratio + 1)
    3) Pick the first point in a batch to have the minority sample, and the second point to have the majority sample.
    4) Therefore, among the (batch size - 2) remaining points in the batch, we expect to see 
       batch size * 1 / (imbalance ratio + 1) - 1 samples with the minority label, given the dataset's class distribution.
    5) Then, to maintain the same label distribution as the full dataset, if we set the probability that any successive
       points (after the second) belong to the minority class to:
       (batch size / (imbalance ratio + 1) - 1) / (batch size - 2),
       then the expected number of minority samples in the batch matches the full-dataset class distribution.

    Finally, if at any point, the number of minority class points left to be sampled is equal to the number of batches that
    we still must create, then force each successive batch to contain exactly one minority class point (such that every batch
    contain at least one minority-label point).
    """

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool, shuffle: bool = True, seed: int = 0) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rand = random.Random(seed)
        self.pos_sampler = SingleLabelSampler(dataset, label_to_keep=1, shuffle=shuffle, rand=self.rand)
        self.neg_sampler = SingleLabelSampler(dataset, label_to_keep=0, shuffle=shuffle, rand=self.rand)
        neg_pos_ratio = len(self.neg_sampler) / len(self.pos_sampler)
        pos_is_minority = True
        if neg_pos_ratio > 1.0:
            pos_is_minority = True
            # imbalance_ratio is the ratio of majority class to minority class
            self.imbalance_ratio = neg_pos_ratio
        else:
            pos_is_minority = False
            self.imbalance_ratio = 1.0/neg_pos_ratio
        if self.imbalance_ratio > batch_size - 1:
            raise ValueError(f"Cannot guarantee one minority sample in each batch if the batch size ({batch_size}) is smaller than the majority-minority class imbalance ratio ({round(self.imbalance_ratio, 4)}) minus 1")
        self.subsequent_minority_probability = (self.batch_size / (self.imbalance_ratio + 1.0) - 1) / (self.batch_size - 2)

        if pos_is_minority:
            self.minority_sampler = self.pos_sampler
            self.majority_sampler = self.neg_sampler
        else:
            self.minority_sampler = self.neg_sampler
            self.majority_sampler = self.pos_sampler
    
        self.minority_points_sampled = 0
        self.majority_points_sampled = 0
        self.batches_remaining = len(self)

    def __iter__(self) -> Iterator[List[int]]:
        minority_generator = self.minority_sampler.__iter__()
        majority_generator = self.majority_sampler.__iter__()
        balanced_batch = []
        while True:
            # Place one minority class sample in the batch, and one majority class sample, to guarantee
            # that points of both labels will be seen in the batch.
            minority_idx = next(minority_generator, None)
            majority_idx = next(majority_generator, None)
            # Neither iterator should be empty at this point, by construction.
            assert minority_idx is not None and majority_idx is not None, breakpoint()
            balanced_batch.append(minority_idx)
            self.minority_points_sampled += 1
            balanced_batch.append(majority_idx)
            self.majority_points_sampled += 1
            self.batches_remaining -= 1

            while len(balanced_batch) < self.batch_size and self.num_points_remaining() > 0:
                p = self.rand.random()
                if self.num_minority_points_remaining() == self.batches_remaining:
                    # If we have nearly exhausted all the minority points to distribute, set p = 1.0
                    # to force the rest of the points in this batch to belong to the majority class.
                    p = 1.0
                elif self.num_majority_points_remaining() == self.batches_remaining:
                    # Do the equivalent thing for the majority class, to ensure equal representation.
                    p = 0.0

                if p < self.subsequent_minority_probability:
                    # Place minority sample into batch.
                    sample_idx = next(minority_generator)
                    self.minority_points_sampled += 1
                else:
                    # Place majority sample into batch.
                    sample_idx = next(majority_generator)
                    self.majority_points_sampled += 1
                balanced_batch.append(sample_idx)

            if self.num_points_remaining() <= 1:
                # If there are no points remaining, then we are done creating batches.
                # If there is one point remaining, there is no way to construct a final batch including both
                # class labels, so skip it. Either way, this is the final batch.
                if len(balanced_batch) > 0 and not self.drop_last:
                    self.maybe_shuffle_batch(balanced_batch)
                    yield balanced_batch
                break
            elif len(balanced_batch) == self.batch_size:
                self.maybe_shuffle_batch(balanced_batch)
                yield balanced_batch
                balanced_batch = []

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return int(np.floor((len(self.minority_sampler) + len(self.majority_sampler)) / float(self.batch_size)))  # type: ignore[arg-type]
        else:
            return int(np.ceil((len(self.minority_sampler) + len(self.majority_sampler)) / float(self.batch_size)))  # type: ignore[arg-type]

    def maybe_shuffle_batch(self, batch: List) -> List:
        if self.shuffle:
            self.rand.shuffle(batch)

    def num_minority_points_remaining(self):
        return len(self.minority_sampler) - self.minority_points_sampled

    def num_majority_points_remaining(self):
        return len(self.majority_sampler) - self.majority_points_sampled

    def num_points_remaining(self):
        return self.num_minority_points_remaining() + self.num_majority_points_remaining()

    def num_batches_remaining(self):
        return self.batches_remaining