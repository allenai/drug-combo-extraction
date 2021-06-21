import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoTokenizer
from typing import Dict, List

from constants import CLS, ENTITY_END_MARKER, ENTITY_PAD_IDX, ENTITY_START_MARKER, SEP

def make_fixed_length(array: List, max_length: int, padding_value: int = 0) -> List:
    """Helper function to make a variable-length array into a fixed-length one.
    If the array is shorter than the fixed length, pad with the given value. If
    longer than the fixed length, truncate it.

    Args:
        array: Array whose length we want to fix
        max_length: Desired length to fix
        padding_value: Value to pad array with, if shorter than desired length

    Returns:
        fixed_array: Fixed-length, padded version of the input array.
    """
    if len(array) >= max_length:
        fixed_array = array[:max_length]
    else:
        pad_length = max_length - len(array)
        fixed_array = array + [padding_value] * pad_length
    return fixed_array

def construct_dataset(data: List[Dict], tokenizer: AutoTokenizer, max_seq_length: int = 512) -> TensorDataset:
    """Converts raw data (in the form of text/label pairs) into a binarized, training-ready Torch TensorDataset.

    Args:
        data: List of dictionaries, each containing a string of entity-marked text and a discrete label
        tokenizer: Huggingface tokenizer, to perform word segmentation
        max_seq_length: Fixed length (in subwords) to use for representing all documents

    Returns:
        dataset: TensorDataset containing numerical representation of the dataset's text strings and discrete labels.
    """
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    targets = []
    all_entity_idxs = []

    max_entities_length = -1
    # Store subwords and entity positions for each document in the first pass over the dataset.
    all_doc_subwords = []
    all_doc_entity_start_positions = []
    for doc in data:
        targets.append(doc["target"])
        doc_subwords = [CLS]
        whitespace_tokens = doc["text"].split()
        entity_start_token_idxs = []

        # Manually split up each token into subwords, to directly identify special entity tokens
        # and store their locations.
        for token in whitespace_tokens:
            if token == ENTITY_START_MARKER:
                entity_start_idx = len(doc_subwords)
                entity_start_token_idxs.append(entity_start_idx)
                doc_subwords.append(ENTITY_START_MARKER)
            elif token == ENTITY_END_MARKER:
                doc_subwords.append(ENTITY_END_MARKER)
            else:
                # If not a special token, then split the token into subwords.
                for sub_token in tokenizer.tokenize(token):
                    doc_subwords.append(sub_token)
        doc_subwords.append(SEP)
        all_doc_subwords.append(doc_subwords)
        all_doc_entity_start_positions.append(entity_start_token_idxs)
        max_entities_length = max(max_entities_length, len(entity_start_token_idxs))

    for i, doc_subwords in enumerate(all_doc_subwords):
        doc_input_ids = tokenizer.convert_tokens_to_ids(doc_subwords)
        entity_start_token_idxs = all_doc_entity_start_positions[i]
        attention_mask = [1] * len(doc_input_ids)
        # TODO(Vijay): figure out why this field is necessary and used in the PURE model.
        segment_ids = [0] * len(doc_input_ids)
        all_input_ids.append(make_fixed_length(doc_input_ids, max_seq_length))
        all_token_type_ids.append(make_fixed_length(segment_ids, max_seq_length))
        all_attention_masks.append(make_fixed_length(attention_mask, max_seq_length))
        all_entity_idxs.append(make_fixed_length(entity_start_token_idxs, max_entities_length, padding_value=ENTITY_PAD_IDX))

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    all_entity_idxs = torch.tensor(all_entity_idxs, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_masks, targets, all_entity_idxs)
    return dataset

class DrugSynergyDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: List[Dict],
                 test_data: List[Dict],
                 tokenizer: AutoTokenizer,
                 label_to_idx: Dict,
                 train_batch_size: int = 32,
                 dev_batch_size: int = 32,
                 test_batch_size: int = 32,
                 dev_train_ratio: float = 0.1,
                 max_seq_length: int = 512,
                 num_workers: int = 4):
        '''Construct a DataModule for convenient PyTorchLightning training.

        Args:
            train_data: List of (text, label) pairs for training and validation
            test_data: List of (text, label) pairs for testing
            tokenizer: Tokenizer/subword segmenter to process raw text
            label_to_idx: Fixed mapping of label strings to numerical values
            train_batch_size: Batch size for training
            dev_batch_size: Batch size for validation
            test_batch_size: Batch size for testing
            dev_train_ratio: Hold out this fraction of the training set as a dev set
            max_seq_length: Fixed document length to use for the dataset
            num_workers: Number of CPU workers to use for loading data

        Returns:
            self: PyTorchLightning DataModule to load all data during training, validation, and testing.
        '''
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.test_batch_size = test_batch_size
        self.dev_train_ratio = dev_train_ratio
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # TODO(Vijay): set dimensions here
        self.dims = (1, 28, 28)

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        full_dataset = construct_dataset(self.train_data, self.tokenizer, max_seq_length=self.max_seq_length)
        dev_size = int(self.dev_train_ratio * len(full_dataset))    
        train_size = len(full_dataset) - dev_size
        self.train, self.val = random_split(full_dataset, [train_size, dev_size])
        self.test = construct_dataset(self.test_data, self.tokenizer, max_seq_length=self.max_seq_length)
        # Optionally...
        # self.dims = tuple(self.train[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.dev_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, num_workers=self.num_workers)