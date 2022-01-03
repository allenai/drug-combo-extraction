'''
Usage:
python dataset_conversion/convert_ddi.py \
    --ddi-dir dataset_conversion/DDICorpus \
    --out-dir dataset_conversion/DDICorpus/synergy_format
'''

import argparse
import os
import re
import sys
sys.path.extend(["..", "."])
from tqdm import tqdm
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET

import common.utils as utils
from convert_scierc import accumulate_relation_labels
from utils.tokenizer import make_tok_seg

tok_seg = make_tok_seg()

parser = argparse.ArgumentParser()
parser.add_argument('--ddi-dir', type=str, required=False, default="dataset_conversion/DDICorpus", help="Path to root directory for the DDI corpus")
parser.add_argument('--out-dir', type=str, required=False, default="dataset_conversion/DDICorpus/synergy_format", help="Directory to place DDI dataset, formatted into the drug synergy format")

def get_xml_files(parent_dir, split):
    dir = os.path.join(parent_dir, split)
    xml_files = []
    
    if split == "Train":
        parent_dirs = []
        for subdir in os.listdir(dir):
            if not subdir.startswith('.'):
                parent_dirs.append(os.path.join(dir, subdir))
    elif split == "Test":
        parent_dirs = []
        for parent_dir in os.listdir(dir):
            if parent_dir.startswith('.'):
                continue
            for subdir in os.listdir(os.path.join(dir, parent_dir)):
                if not subdir.startswith('.'):
                    parent_dirs.append(os.path.join(dir, parent_dir, subdir))
    for dir in parent_dirs:
        if dir.startswith('.'):
            continue
        for f in os.listdir(dir):
            if f.endswith(".xml"):
                xml_files.append(os.path.join(dir, f))
    return xml_files

def split_tokens_from_whitespace(str):
    return re.split("(\s+)", str)

def char_to_token_mapping(sentence):
    start_char_to_token = {}
    end_char_to_token = {}
    tokens_and_whitespace = split_tokens_from_whitespace(sentence)
    char_counter = 0
    for i, token in enumerate(tokens_and_whitespace):
        start_counter = char_counter
        char_counter += len(token)
        end_counter = char_counter
        if len(token.split()) == 0:
            continue
        start_char_to_token[start_counter] = i
        end_char_to_token[end_counter] = i
    return start_char_to_token, end_char_to_token

def retokenize_sentence(sentence, entity_idxs):
    position_offsets = []
    truncation_string_start_idx = sentence.find("(ABSTRACT TRUNCATED")
    if truncation_string_start_idx != -1:
        sentence = sentence[:truncation_string_start_idx] + " " + sentence[truncation_string_start_idx:]
        position_offsets.append((truncation_string_start_idx, 1))

    retokenized = ""
    tokens_and_whitespace = split_tokens_from_whitespace(sentence)
    original_char_counter = 0
    for i, token in enumerate(tokens_and_whitespace):
        if len(token.split()) > 0:
            subtokens = list(tok_seg(token))
            if len(subtokens) == 1:
                retokenized += token
                original_char_counter += len(token)
            else:
                for i, subtoken_tok in enumerate(subtokens):
                    subtoken = subtoken_tok.text
                    retokenized += subtoken
                    offsetted_length = len(retokenized)
                    subtoken_in_entity = False
                    for entity in entity_idxs:
                        if original_char_counter + len(subtoken) >= entity[1] and original_char_counter + len(subtoken) <= entity[2] and \
                            i + 1 < len(subtokens) and original_char_counter + len(subtoken) + len(subtokens[i+1]) <= entity[2]:
                            # TODO(Vijay): ensure this end-index is correct.
                            subtoken_in_entity = True
                    original_char_counter += len(subtoken)
                    if not subtoken_in_entity and i < len(subtokens) - 1:
                        retokenized += " "
                        previous_offset = sum([offset for j, offset in position_offsets if j < original_char_counter])
                        position_offsets.append((offsetted_length - previous_offset, 1))
        else:
            retokenized += token
            original_char_counter += len(token)

    for _, entity_start, entity_end in entity_idxs:
        offsetted_start = entity_start + sum([offset for j, offset in position_offsets if j < entity_start])
        if offsetted_start > 0 and retokenized[offsetted_start - 1] != " ":
            retokenized = retokenized[:offsetted_start] + " " + retokenized[offsetted_start:]
            position_offsets.append((entity_start - 1, 1))
        offsetted_end = entity_end + sum([offset for j, offset in position_offsets if j < entity_end])
        if offsetted_end < len(retokenized) and retokenized[offsetted_end] != " ":
            retokenized = retokenized[:offsetted_end] + " " + retokenized[offsetted_end:]
            position_offsets.append((entity_end, 1))
    return retokenized, position_offsets

def get_char_indices(entity):
    if ";" in entity.get('charOffset'):
        chunks = entity.get('charOffset').split(';')
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        char_start_str = first_chunk.split('-')[0]
        char_end_str = last_chunk.split('-')[1]
    else:
        char_start_str, char_end_str = entity.get('charOffset').split('-')
    char_start, char_end = int(char_start_str), int(char_end_str)+1
    return char_start, char_end

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    assert root.tag == "document"
    rows = []
    sentences = []
    for child_sentence in root:
        sentence_id = child_sentence.get('id')
        sentence_idx = int(sentence_id.split('.')[-1][1:])
        assert child_sentence.tag == "sentence"
        sentence = child_sentence.get('text')

        entity_idxs = []
        for entity in child_sentence:
            assert entity.tag in ["entity", "pair"]
            if entity.tag != "entity":
                continue
            char_start, char_end = get_char_indices(entity)
            entity_idxs.append((entity.get('text'), char_start, char_end))


        retokenized_sentence, position_offsets = retokenize_sentence(sentence, entity_idxs)
        start_char_to_token, end_char_to_token = char_to_token_mapping(retokenized_sentence)


        converted_spans = []
        entity_id_to_span_id = {}
        skip_sentence = False
        for entity in child_sentence:
            assert entity.tag in ["entity", "pair"]
            if entity.tag != "entity":
                continue
            char_start, char_end = get_char_indices(entity)
            char_start += sum([offset for idx, offset in position_offsets if idx <= char_start])
            char_end += sum([offset for idx, offset in position_offsets if idx < char_end])

            if char_start not in start_char_to_token or char_end not in end_char_to_token:
                skip_sentence = True
                break
            token_start = start_char_to_token[char_start]
            token_end = end_char_to_token[char_end]

            span_id = len(converted_spans)
            span = {
                "span_id": span_id,
                "text": entity.get('text'),
                "start": char_start,
                "end": char_end,
                "token_start": token_start,
                "token_end": token_end
            }
            entity_id_to_span_id[entity.get('id')] = span_id
            converted_spans.append(span)
        if skip_sentence:
            converted_spans = []
            continue

        converted_relations = []
        if not skip_sentence:
            for relation in child_sentence:
                assert relation.tag in ["entity", "pair"]
                if relation.tag != "pair":
                    continue
                entity_ids = [relation.get('e1'), relation.get('e2')]
                span_ids = []
                for entity_id in entity_ids:
                    relation_sentence_idx = int(entity_id.split('.')[-2][1:])
                    assert relation_sentence_idx == sentence_idx
                    span_ids.append(entity_id_to_span_id[entity_id])
                converted_relation = {"class": relation.get('ddi'), "spans": span_ids}
                converted_relations.append(converted_relation)

        if len(converted_relations) > 0:
            converted_row = {
                "sentence": sentence,
                "spans": converted_spans,
                "rels": converted_relations,
                "paragraph": None,
            }
            rows.append(converted_row)
        sentences.append(sentence)
    paragraph = " ".join(sentences)
    for i in range(len(rows)):
        rows[i]["paragraph"] = paragraph
    return rows

def load_relation_annotations(xml_files):
    rows = []
    for i, f in tqdm(enumerate(xml_files)):
        rows.extend(parse_xml(f))
    return rows

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    splits = ["Train", "Test"]

    train_xml_files = get_xml_files(args.ddi_dir, "Train")
    train_rows = load_relation_annotations(train_xml_files)

    test_xml_files = get_xml_files(args.ddi_dir, "Test")
    test_rows = load_relation_annotations(test_xml_files)

    label2idx = accumulate_relation_labels(train_rows + test_rows)
    utils.write_json(label2idx, os.path.join(args.out_dir, "label2idx.json"))
    utils.write_jsonl(train_rows, os.path.join(args.out_dir, "train.jsonl"))
    utils.write_jsonl(test_rows, os.path.join(args.out_dir, "test.jsonl"))