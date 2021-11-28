'''
Usage:
python convert_scierc.py --merge-dev-and-train
'''

import argparse
import os
import sys
from typing import List, Dict, Tuple

import common.utils as utils

BASIC_PUNCTUATION = [".", "!", "?", ",", ";", ":"]

parser = argparse.ArgumentParser()
parser.add_argument('--scierc-dir', type=str, required=False, default="sciERC/processed_data/json", help="Path to processed_data directory for SciERC dataset")
parser.add_argument('--out-dir', type=str, required=False, default="sciERC/synergy_format", help="Directory to place sciERC dataset, formatted into the drug synergy format")
parser.add_argument('--merge-dev-and-train', action="store_true", help="Whether to merge dev and train splits into a single train split")

def detokenize_sentence(tokens: List[str], token_index_offset: int) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """Converts a list of tokens into a sentence, which can be tokenized later by applications.
    To comply with this, update entity span indices to be span character indices instead of token indices.

    Args:
        tokens: List of tokens (representing a tokenized sentence)
        token_index_offset: Number of tokens prior to this sentence in the containing paragraph, because our token-character index
                            mapping must correspond to positions in the entire paragraph.
        char_index_offset: Number of characters prior to this sentence in the containing paragraph.

    Returns:
        detokenized_sentence: Single string representing the detokenized sentence.
        token_char_mapping: Mapping from paragraph-level token indices to character indices for each token in the sentence.
    """

    detokenized_sentence = ""
    token_char_mapping = {}
    token_start_idx = 0
    for i, token in enumerate(tokens):
        paragraph_idx = token_index_offset + i
        token_end_idx = token_start_idx + len(token)
        token_char_mapping[paragraph_idx] = (token_start_idx, token_end_idx)
        detokenized_sentence = detokenized_sentence + token + " "
        token_start_idx += len(token) + 1
    detokenized_sentence = detokenized_sentence[:-1]
    return detokenized_sentence, token_char_mapping


def update_entity_indices(entities: List[List], sentence: str, paragraph_tokens: List[str], token_index_mapping: Dict[int, Tuple[int, int]], entities_seen: int, token_offset: int) -> List[Dict]:
    """Convert SciERC entity objects within a sentence into Drug Synergy Dataset entity span objects. In addition to
    converting the data format, also update all token spans in each entity to be paragraph-level character spans.

    Args:
        entities: SciERC entity annotations within a single sentence
        sentence: String containing full text in the detokenized sentence
        sentence_tokens: Tokens in the sentence
        token_index_mapping: Mapping from the index of each token to its start and end character indices (to construct character spans)
        entities_seen: Number of entities observed in previous sentences (for the purposes of constructing unique entity IDs)
        token_offset: Number of tokens observed in previous sentences

    Returns:
        converted_spans: SciERC entity annotations within a sentence, converted into the Drug Synergy Dataset format
    """
    converted_spans = []

    for i, entity in enumerate(entities):
        [start_token_idx, end_token_idx, entity_type] = entity
        start_char_idx, _ = token_index_mapping[start_token_idx]
        _, end_char_idx = token_index_mapping[end_token_idx]
        text = sentence[start_char_idx:end_char_idx]

        word = paragraph_tokens[start_token_idx:end_token_idx+1]
        if len(word) == 1:
            # As a sanity check, verify that all single token entities match perfectly here
            # between the token-indexed and char-indexed entities.
            assert word[0] == text, breakpoint()

        token_start_string = text.split()[0]
        token_end_string = text.split()[-1]
        span = {
            "span_id": i + entities_seen,
            "text": text,
            "start": start_char_idx,
            "end": end_char_idx,
            "token_start": start_token_idx - token_offset,
            "token_end": end_token_idx - token_offset
        }
        converted_spans.append(span)
    return converted_spans

def update_relation_indices(relations: List[List], entities: List[Dict], token_index_mapping: Dict[int, Tuple[int, int]]) -> List[Dict]:
    """Convert SciERC relation annotations within a sentence into Drug Synergy Dataset relation objects. While SciERC
    represents the entities in each span by their token indices, the Synergy Dataset refers to each entity by its index in the entities list.

    Args:
        relations: SciERC relation annotations within a sentence
        entities: SciERC entities (converted into Synergy Dataset format) within a sentence
        token_index_mapping: Mapping from the index of each token to its start and end character indices (to construct character spans)

    Returns:
        converted_relations: SciERC relation annotations within a sentence, converted into the Drug Synergy Dataset format
    """
    converted_relations = []
    for relation in relations:
        if len(relation) == 0:
            continue
        entity_a_token_start, entity_a_token_end, entity_b_token_start, entity_b_token_end, relation_class = relation
        entity_a_char_start = token_index_mapping[entity_a_token_start][0]
        entity_a_char_end = token_index_mapping[entity_a_token_end][1]
        entity_b_char_start = token_index_mapping[entity_b_token_start][0]
        entity_b_char_end = token_index_mapping[entity_b_token_end][1]

        # Match the entity character indices to the `entities` list.        
        spans = []
        for entity_span in [(entity_a_char_start, entity_a_char_end), (entity_b_char_start, entity_b_char_end)]:
            entity_matched = False
            for i, entity in enumerate(entities):
                marker = 2
                if entity["start"] == entity_span[0] and entity["end"] == entity_span[1]:
                    spans.append(i)
                    entity_matched = True
                    break
            assert entity_matched is True
        converted_relation = {"class": relation_class, "spans": spans}
        converted_relations.append(converted_relation)
    return converted_relations


def convert_scierc_rows(row: Dict) -> Dict:
    """Convert a single row from SciERC (containing a document with entities and relations annotated at each sentence)
    into a set of Drug Synergy rows (each containing a sentence with entities and relations).

    Args:
        row: Raw annotated SciERC document

    Returns:
        rows_converted: SciERC document converted into a list of sentence-level Synergy Dataset rows
    """
    paragraph = ""
    token_index_offset = 0
    token_index_mapping = {}
    sentences = []

    # In a first pass of the paragraph, convert each entity span (originally represented as a span of tokens)
    # into a span of characters, indexed over all characters in the paragraph.
    all_tokens = []
    token_offsets = [0]
    for i, sentence in enumerate(row["sentences"]):
        detokenized_sentence, sentence_token_index_mapping = detokenize_sentence(sentence, token_index_offset)
        sentences.append(detokenized_sentence)
        token_index_mapping.update(sentence_token_index_mapping)
        paragraph = paragraph + detokenized_sentence + " "
        all_tokens.extend(sentence)
        token_index_offset += len(sentence)
        token_offsets.append(token_index_offset)
    token_offsets = token_offsets[:-1]

    paragraph = paragraph[:-1] # remove trailing space from the paragraph text.
    rows_converted = []
    entities_seen = 0
    for i in range(len(row["sentences"])):
        sentence = sentences[i]
        converted_entities = update_entity_indices(row["ner"][i], sentences[i], all_tokens, token_index_mapping, entities_seen, token_offsets[i])
        converted_relations = update_relation_indices(row["relations"][i], converted_entities, token_index_mapping)
        converted_row = {
            "sentence": sentence,
            "spans": converted_entities,
            "rels": converted_relations,
            "paragraph": paragraph,
        }
        if len(converted_relations) == 0:
            # Ignore sentences with no relations annotated.
            continue
        rows_converted.append(converted_row)
        entities_seen += len(converted_entities)
    return rows_converted

def convert_scierc_split(split_data: List[Dict]) -> List[Dict]:
    """Convert a split of SciERC data (dev, train, or test) into Drug Synergy Dataset format.

    Args:
        split_data: SciERC split

    Returns:
        converted_rows: SciERC split converted into Synergy Dataset format
    """
    converted_rows = []
    for raw_row in split_data:
        converted_rows.extend(convert_scierc_rows(raw_row))
    return converted_rows

def accumulate_relation_labels(dataset: List[Dict]) -> List[Dict]:
    label2idx = {}
    for document in dataset:
        for relation in document["rels"]:
            relation_label = relation["class"]
            if relation_label not in label2idx:
                label2idx[relation_label] = len(label2idx) + 1
    return label2idx

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    splits = ["train", "test", "dev"]

    dev_train_set = []
    full_data = []
    for split in splits:
        in_file = os.path.join(args.scierc_dir, split + ".json")
        raw_split_data = utils.read_jsonl(in_file)
        converted_split = convert_scierc_split(raw_split_data)
        out_file = os.path.join(args.out_dir, split + ".jsonl")
        if args.merge_dev_and_train:
            # This option allows the construction of a single joint dev-train set, if desired.
            if split == "train":
                dev_train_set.extend(converted_split)
            elif split == "dev":
                dev_train_set.extend(converted_split)
        utils.write_jsonl(converted_split, out_file)
        full_data.extend(converted_split)

    label2idx = accumulate_relation_labels(full_data)
    utils.write_json(label2idx, os.path.join(args.out_dir, "label2idx.json"))

    if args.merge_dev_and_train:
        utils.write_jsonl(dev_train_set, os.path.join(args.out_dir, "dev_train.jsonl"))