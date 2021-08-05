from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoTokenizer
from typing import Any, Dict, List, Tuple

from balanced_batch_sampler import BalancedBatchSampler
from constants import CLS, COREF_PAD_IDX, ENTITY_END_MARKER, ENTITY_PAD_IDX, ENTITY_START_MARKER, SEP
from utils import pad_lower_right, tuple_max, separate_tokens_from_whitespace, rejoin_tokens_and_whitespaces

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

def match_index_to_range_dict(index: int, range_dict: Dict) -> Any:
    '''
    Args:
        index: value to look up in the range dict
        range_dict: dictionary where each key points to a range of values
    '''
    for (range_start, range_end), value in range_dict.items():
        if index >= range_start and index < range_end:
            return value
    raise ValueError("Index not matched to any range")

def tokenize_sentence(text: str, tokenizer: AutoTokenizer, ner_spans: List[List], coreference_clusters: List[List]) -> Tuple[List[str], List[int]]:
    '''Given a text sentence, run the Huggingface subword tokenizer on this sentence,
    and return a list of subword tokens and the positions of all special entity marker
    tokens in the text.

    Args:
        text: String to tokenize
        tokenizer: HuggingFace tokenizer
        coreference_clusters: List containing clusters of coreferring entities

    Returns:
        doc_subwords: List of subword strings
        entity_start_token_idxs: Positions of all entity-start tokens in the list of subwords
    '''
    doc_subwords = [CLS]
    tokens, whitespaces = separate_tokens_from_whitespace(text)
    entity_start_token_idxs = []
    entity_end_token_idxs = []

    # Manually split up each token into subwords, to directly identify special entity tokens
    # and store their locations.
    char_subword_idx_mapping = {}
    cur_char_idx = 0
    for token_idx, token in enumerate(tokens):
        if token == ENTITY_START_MARKER or token == ENTITY_END_MARKER:
            char_subword_idx_mapping[(cur_char_idx, cur_char_idx + len(token))] = (len(doc_subwords), len(doc_subwords) + 1)
            if token == ENTITY_START_MARKER:
                entity_start_token_idxs.append(len(doc_subwords))
            else:
                entity_end_token_idxs.append(len(doc_subwords))
            doc_subwords.append(token)
        else:
            token_subwords = tokenizer.tokenize(token)
            char_subword_idx_mapping[(cur_char_idx, cur_char_idx + len(token))] = (len(doc_subwords), len(doc_subwords) + len(token_subwords))
            for sub_token in token_subwords:
                doc_subwords.append(sub_token)
        num_trailing_whitespaces = 0 if token_idx >= len(whitespaces) else len(whitespaces[token_idx])
        cur_char_idx += len(token) + num_trailing_whitespaces
    doc_subwords.append(SEP)

    token_indexed_ner_span_indices = []
    for span_start_idx, span_end_idx in ner_spans:
        span_start_token_idx, _ = match_index_to_range_dict(span_start_idx, char_subword_idx_mapping)
        _, span_end_token_idx = match_index_to_range_dict(span_end_idx-1, char_subword_idx_mapping)
        matched = False
        for t in token_indexed_ner_span_indices:
            if t == [span_start_token_idx, span_end_token_idx]:
                matched = True
        if not matched:
            token_indexed_ner_span_indices.append([span_start_token_idx, span_end_token_idx])

    token_indexed_coreference_clusters = []
    for cluster_coreference_indices in coreference_clusters:
        token_indexed_cluster_coreference_indices = []
        for span_start_idx, span_end_idx in cluster_coreference_indices:
            span_start_token_idx, _ = match_index_to_range_dict(span_start_idx, char_subword_idx_mapping)
            _, span_end_token_idx = match_index_to_range_dict(span_end_idx-1, char_subword_idx_mapping)
            token_indexed_cluster_coreference_indices.append([span_start_token_idx, span_end_token_idx])
        token_indexed_coreference_clusters.append(token_indexed_cluster_coreference_indices)

    return doc_subwords, entity_start_token_idxs, entity_end_token_idxs, token_indexed_ner_span_indices, token_indexed_coreference_clusters

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

def generate_coreference_matrix(span_indices: List[List], span_coreferences: List[List[List]]) -> np.array:
    span_index_lookup = {}
    for i, span in enumerate(span_indices):
        assert tuple(span) not in span_index_lookup, breakpoint()
        span_index_lookup[tuple(span)] = i
    coreference_matrix = np.zeros((len(span_index_lookup), len(span_index_lookup)))
    for coreference_cluster in span_coreferences:
        for span_1 in coreference_cluster:
            assert tuple(span_1) in span_index_lookup
            span_1_index = span_index_lookup[tuple(span_1)]
            for span_2 in coreference_cluster:
                span_2_index = span_index_lookup[tuple(span_2)]
                coreference_matrix[span_1_index][span_2_index] = 1.0
    return coreference_matrix

def construct_span_pair_relation_matrix(span_coreferences: List) -> List[List]:
    pass

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
    all_doc_entity_end_positions = []
    all_row_ids = []
    all_span_indices = []
    all_coref_matrices = []

    max_dimensions_span_indices = None
    max_dimensions_coref_matrices = None
    for i, doc in enumerate(tqdm(data)):
        targets.append(doc["target"])
        doc_subwords, entity_start_token_idxs, entity_end_token_idxs, span_indices, span_coreferences = tokenize_sentence(doc["text"], tokenizer, doc["ner_spans"], doc["coreference_clusters"])
        all_doc_subwords.append(doc_subwords)
        all_doc_entity_start_positions.append(entity_start_token_idxs)
        all_doc_entity_end_positions.append(entity_end_token_idxs)
        max_entities_length = max(max_entities_length, len(entity_end_token_idxs))
        all_row_ids.append(row_idx_mapping[doc["row_id"]])
        span_indices_matrix = np.array(span_indices)
        all_span_indices.append(span_indices_matrix)
        max_dimensions_span_indices = tuple_max(max_dimensions_span_indices, span_indices_matrix.shape)
        coref_matrix = generate_coreference_matrix(span_indices, span_coreferences)
        all_coref_matrices.append(coref_matrix)
        max_dimensions_coref_matrices = tuple_max(max_dimensions_coref_matrices, coref_matrix.shape)

    assert max_dimensions_span_indices is not None
    assert max_dimensions_coref_matrices is not None

    entity_start_idxs = []
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []

    all_span_indices_padded = []
    all_coref_matrices_padded = []

    for i, doc_subwords in enumerate(all_doc_subwords):
        entity_start_token_idxs = all_doc_entity_start_positions[i]
        entity_start_idxs.append(make_fixed_length(entity_start_token_idxs, max_entities_length, padding_value=ENTITY_PAD_IDX))

        row = vectorize_subwords(tokenizer, doc_subwords, max_seq_length)
        all_input_ids.append(row.input_ids)
        all_token_type_ids.append(row.segment_ids)
        all_attention_masks.append(row.attention_mask)

        padded_span_indices = pad_lower_right(all_span_indices[i], max_dimensions_span_indices, pad_value=COREF_PAD_IDX)
        padded_coref_matrix = pad_lower_right(all_coref_matrices[i], max_dimensions_coref_matrices, pad_value=COREF_PAD_IDX)

        all_span_indices_padded.append(padded_span_indices.tolist())
        all_coref_matrices_padded.append(padded_coref_matrix.tolist())

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    entity_start_idxs = torch.tensor(entity_start_idxs, dtype=torch.long)
    all_row_ids = torch.tensor(all_row_ids, dtype=torch.long)
    all_span_indices_padded = torch.tensor(all_span_indices_padded, dtype=torch.long)
    all_coref_matrices_padded = torch.tensor(all_coref_matrices_padded, dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
                            all_token_type_ids,
                            all_attention_masks,
                            targets,
                            entity_start_idxs,
                            all_row_ids,
                            all_span_indices_padded,
                            all_coref_matrices_padded)
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