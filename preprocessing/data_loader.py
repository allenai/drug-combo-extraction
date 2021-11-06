import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

from preprocessing.balanced_batch_sampler import BalancedBatchSampler
from common.constants import CLS, ENTITY_END_MARKER, ENTITY_START_MARKER, SEP

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

def tokenize_sentence(text: str, tokenizer: AutoTokenizer, tokenizer_cache: Dict) -> Tuple[List[str], List[int]]:
    '''Given a text sentence, run the Huggingface subword tokenizer on this sentence,
    and return a list of subword tokens and the positions of all special entity marker
    tokens in the text.

    Args:
        text: String to tokenize
        tokenizer: HuggingFace tokenizer

    Returns:
        doc_subwords: List of subword strings
        entity_start_token_idxs: Positions of all entity-start tokens in the list of subwords
    '''
    doc_subwords = [CLS]
    whitespace_tokens = text.split()
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
            if token in tokenizer_cache:
                sub_tokens = tokenizer_cache[token]
            else:
                sub_tokens = tokenizer.tokenize(token)
                tokenizer_cache[token] = sub_tokens
            for sub_token in sub_tokens:
                doc_subwords.append(sub_token)
    doc_subwords.append(SEP)
    return doc_subwords, entity_start_token_idxs

class DatasetRow:
    def __init__(self, input_ids, attention_mask, segment_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids

def vectorize_subwords(tokenizer, doc_subwords: List[str], max_seq_length: int = 512):
    doc_input_ids = tokenizer.convert_tokens_to_ids(doc_subwords)
    input_ids = make_fixed_length(doc_input_ids, max_seq_length)
    attention_mask = make_fixed_length([1] * len(doc_input_ids), max_seq_length)
    # Treat entire paragraph as a single segment, without SEP tokens
    segment_ids = make_fixed_length([0] * len(doc_input_ids), max_seq_length)
    return DatasetRow(input_ids, attention_mask, segment_ids)

def construct_dataset(data: List[Dict], tokenizer: AutoTokenizer, row_idx_mapping: Dict, max_seq_length: int = 512) -> TensorDataset:
    """Converts raw data (in the form of text/label pairs) into a binarized, training-ready Torch TensorDataset.

    Args:
        data: List of dictionaries, each containing a string of entity-marked text and a discrete label
        tokenizer: Huggingface tokenizer, to perform word segmentation
        row_idx_mapping: Maps each unique row identifier to an integer.
        max_seq_length: Fixed length (in subwords) to use for representing all documents

    Returns:
        dataset: TensorDataset containing numerical representation of the dataset's text strings and discrete labels.
    """
    targets = []
    max_entities_length = -1
    # Store subwords and entity positions for each document in the first pass over the dataset.
    all_doc_subwords = []
    all_doc_entity_start_positions = []
    all_row_ids = []
    tokenizer_cache = {}
    for doc in tqdm(data):
        targets.append(doc["target"])
        doc_subwords, entity_start_token_idxs = tokenize_sentence(doc["text"], tokenizer, tokenizer_cache)
        all_doc_subwords.append(doc_subwords)
        all_doc_entity_start_positions.append(entity_start_token_idxs)
        max_entities_length = max(max_entities_length, len(entity_start_token_idxs))
        all_row_ids.append(row_idx_mapping[doc["row_id"]])

    all_entity_idx_weights = [] # Used to compute an average of embeddings for the document.
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []

    for i, doc_subwords in enumerate(all_doc_subwords):
        entity_start_token_idxs = all_doc_entity_start_positions[i]
        entity_idx_weights = np.zeros((1, max_seq_length))
        for start_token_idx in entity_start_token_idxs:
            assert start_token_idx < max_seq_length, "Entity is out of bounds in truncated text seqence, make --max-seq-length larger"
            entity_idx_weights[0][start_token_idx] = 1.0/len(entity_start_token_idxs)
        all_entity_idx_weights.append(entity_idx_weights.tolist())
        row = vectorize_subwords(tokenizer, doc_subwords, max_seq_length)
        all_input_ids.append(row.input_ids)
        all_token_type_ids.append(row.segment_ids)
        all_attention_masks.append(row.attention_mask)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    all_entity_idx_weights = torch.tensor(all_entity_idx_weights, dtype=torch.float32)
    all_row_ids = torch.tensor(all_row_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_masks, targets, all_entity_idx_weights, all_row_ids)
    return dataset

class DrugSynergyDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: List[Dict],
                 test_data: List[Dict],
                 tokenizer: AutoTokenizer,
                 label_to_idx: Dict,
                 row_idx_mapping: Dict,
                 train_batch_size: int = 32,
                 dev_batch_size: int = 32,
                 test_batch_size: int = 32,
                 dev_train_ratio: float = 0.1,
                 max_seq_length: int = 512,
                 num_workers: int = 4,
                 balance_training_batch_labels: bool = True):
        '''Construct a DataModule for convenient PyTorch Lightning training.

        Args:
            train_data: List of (text, label) pairs for training and validation
            test_data: List of (text, label) pairs for testing
            tokenizer: Tokenizer/subword segmenter to process raw text
            label_to_idx: Fixed mapping of label strings to numerical values
            row_idx_mapping: Maps each unique row identifier to an integer.
            train_batch_size: Batch size for training
            dev_batch_size: Batch size for validation
            test_batch_size: Batch size for testing
            dev_train_ratio: Hold out this fraction of the training set as a dev set
            max_seq_length: Fixed document length to use for the dataset
            num_workers: Number of CPU workers to use for loading data

        Returns:
            self: PyTorch Lightning DataModule to load all data during training, validation, and testing.
        '''
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.row_idx_mapping = row_idx_mapping
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.test_batch_size = test_batch_size
        self.dev_train_ratio = dev_train_ratio
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.balance_training_batch_labels = balance_training_batch_labels

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # TODO(Vijay): set dimensions here
        self.dims = (1, 28, 28)

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        if self.train_data is not None:
            full_dataset = construct_dataset(self.train_data, self.tokenizer, self.row_idx_mapping, max_seq_length=self.max_seq_length)
            dev_size = int(self.dev_train_ratio * len(full_dataset))
            train_size = len(full_dataset) - dev_size
            self.train, self.val = random_split(full_dataset, [train_size, dev_size])
        else:
            self.train, self.val = None, None

        self.test = construct_dataset(self.test_data, self.tokenizer, self.row_idx_mapping, max_seq_length=self.max_seq_length)
        # Optionally...
        # self.dims = tuple(self.train[0][0].shape)

    def train_dataloader(self):
        if self.train is None:
            return None
        if self.balance_training_batch_labels:
            train_batch_sampler = BalancedBatchSampler(dataset = self.train, batch_size = self.train_batch_size, drop_last=False)
            return DataLoader(self.train, num_workers=self.num_workers, batch_sampler=train_batch_sampler)
        else:
            return DataLoader(self.train, num_workers=self.num_workers, batch_size=self.train_batch_size)

    def val_dataloader(self):
        if self.val is None:
            return None
        return DataLoader(self.val, batch_size=self.dev_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, num_workers=self.num_workers)