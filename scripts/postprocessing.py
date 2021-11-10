import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Dict, Tuple
import sys

sys.path.extend(['.', '..'])
from common.constants import ENTITY_PAD_IDX
from modeling.model import BertForRelation
from preprocessing.data_loader import make_fixed_length, tokenize_sentence, vectorize_subwords
from preprocessing.preprocess import add_entity_markers, process_doc_with_unknown_relations, truncate_text_into_window


def extract_all_candidate_relations_for_document(message: Dict,
                       tokenizer: AutoTokenizer,
                       max_seq_length: int,
                       label2idx: Dict,
                       context_window_size: int,
                       include_paragraph_context: bool = True,
                       max_num_candidate_relations: int = 100):
    '''Given a row from the Drug Synergy dataset, find and display all relations with probability greater than some threshold,
    by making multiple calls to the relation classifier.

    Args:
        message: JSON row from the Drug Synergy dataset
        model: Pretrained BertForRelation model object
        tokenizer: Hugging Face tokenizer loaded from disk
        max_seq_length: Maximum number of subwords in a document allowed by the model (if longer, truncate input)
        label2idx: Mapping from label strings to numerical label indices
        label_of_interest: Return relations that maximize the probability of this label (typically, this should be 1 for the POS label)
        include_paragraph_context: Whether or not to include paragraph context in addition to the relation-bearing sentence
    '''
    doc_with_unknown_relations = process_doc_with_unknown_relations(message, label2idx, include_paragraph_context=include_paragraph_context)
    if doc_with_unknown_relations is None or len(doc_with_unknown_relations.relations) > max_num_candidate_relations:
        return None, None
    marked_sentences = []
    relations = []

    tokenizer_cache = {}

    for relation in doc_with_unknown_relations.relations:
        # Mark drug entities with special tokens.
        marked_sentence = add_entity_markers(doc_with_unknown_relations.text, relation.drug_entities)
        marked_sentences.append(truncate_text_into_window(marked_sentence, context_window_size))
        relations.append(tuple(sorted([drug.drug_name for drug in relation.drug_entities])))

    all_entity_idxs = []
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    for sentence in marked_sentences:
        subwords, entity_start_tokens = tokenize_sentence(sentence, tokenizer, tokenizer_cache)
        vectorized_row = vectorize_subwords(tokenizer, subwords, max_seq_length)
        all_input_ids.append(vectorized_row.input_ids)
        all_token_type_ids.append(vectorized_row.attention_mask)
        all_attention_masks.append(vectorized_row.segment_ids)

        entity_idx_weights = np.zeros((1, max_seq_length))
        for start_token_idx in entity_start_tokens:
            if start_token_idx >= max_seq_length:
                return None, None
            entity_idx_weights[0][start_token_idx] = 1.0/len(entity_start_tokens)
        all_entity_idxs.append(entity_idx_weights.tolist())

    all_entity_idxs = torch.tensor(all_entity_idxs, dtype=torch.float32)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    return relations, (all_input_ids, all_token_type_ids, all_attention_masks, all_entity_idxs)

def hash_string(string):
    return hashlib.md5((string).encode()).hexdigest()

def concate_tensors(list_of_tuples):
    tuple_of_lists = [[] for _ in range(len(list_of_tuples[0]))]
    for tup in list_of_tuples:
        for i, val in enumerate(tup):
            tuple_of_lists[i].append(val)
    for i in range(len(tuple_of_lists)):
        tuple_of_lists[i] = torch.cat(tuple_of_lists[i])
    return tuple_of_lists

def find_all_relations(model_inputs: Tuple[torch.Tensor],
                       model: BertForRelation,
                       threshold: float,
                       label_of_interest: int = 1):
    '''Given a row from the Drug Synergy dataset, find and display all relations with probability greater than some threshold,
    by making multiple calls to the relation classifier.

    Args:
        model: Pretrained BertForRelation model object
        threshold: Classifier threshold
        label_of_interest: Return relations that maximize the probability of this label (typically, this should be 1 for the POS label)
    '''
    all_input_ids, all_token_type_ids, all_attention_masks, all_entity_idxs = model_inputs
    logits = model(all_input_ids, token_type_ids=all_token_type_ids, attention_mask=all_attention_masks, all_entity_idxs=all_entity_idxs)
    probability = torch.nn.functional.softmax(logits)
    label_probabilities = probability[:, label_of_interest].tolist()

    relation_probabilities = []
    for i, probability in enumerate(label_probabilities):
        if probability > threshold:
            relation_probabilities.append({"drugs": relations[i], "positive probability": probability})
    relation_probabilities = sorted(relation_probabilities, key=lambda x: x["positive probability"], reverse=True)
    return {'relations': relation_probabilities}
