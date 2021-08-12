'''
python preprocess_pretraining_data.py --in-file /Users/vijay/Downloads/distant_supervision_large.csv \
     --out-directory /Users/vijay/Downloads/distant_supervision_pretraining_data \
     --entities-list drugs.txt
'''

import argparse
import csv
import jsonlines
import os
import re
import random
from tqdm import tqdm
random.seed(2021)

from preprocess import add_entity_markers, DrugEntity
from utils import is_sublist

TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

parser = argparse.ArgumentParser()
parser.add_argument('--in-file', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--entities-list', type=str, required=True)
parser.add_argument('--mask-one-drug-at-a-time', action="store_true")
parser.add_argument('--train-fraction', type=float, default=0.8)
parser.add_argument('--dev-fraction', type=float, default=0.1)

def replace_spans(text, spans, replacement_spans):
    for span, replacement in zip(spans, replacement_spans):
        text = text.replace(span, replacement)
    # Return replacement token indices
    return text

def choose_split(train_fraction, dev_fraction):
    random_value = random.random()
    if random_value <= train_fraction:
        return TRAIN_SPLIT
    elif random_value <= train_fraction + dev_fraction:
        return DEV_SPLIT
    else:
        return TEST_SPLIT

# Taken from https://github.com/aryehgigi/drug-synergy/blob/master/drug_drug_recipe.py#L26-L37
# and adapted slightly.
def find_sent_in_para(sent, para):
    para = para.replace("\u2009", " ").replace("\u00a0", " ").replace("\u202f", " ").replace("\u2003", " ").replace("\u200a", " ")
    idx = para.replace(" ", "").find(sent.replace(" ", ""))
    c = 0
    for i in range(idx):
        while para[i + c] == " ":
            c += 1
    c2 = 0
    for i in range(len(sent.replace(" ", ""))):
        while i + idx + c + c2 < len(para) and para[i + idx + c + c2] == " ":
            c2 += 1
    return idx + c, idx + c + c2 + len(sent.replace(" ", ""))

def is_sublist(list_a, list_b):
    if len(list_a) == 0:
        return True
    for i in range(len(list_b) - len(list_a)):
        if list_b[i] == list_a[0]:
            matched = True
            for j in range(len(list_a) - 1):
                if list_a[j + 1] != list_b[i+j+1]:
                    matched = False
                    break
            if matched:
                return True
    return False

def process_document(sentence, paragraph, drugs_list, mask_one_drug_at_a_time=True, max_arity=10):
    sentence_start_idx, sentence_end_idx = find_sent_in_para(sentence, paragraph)
    paragraph_prefix = paragraph[:sentence_start_idx]
    paragraph_suffix = paragraph[sentence_end_idx:]

    drug_mentions = []
    sentence_lower = sentence.lower()
    sentence_tokens = sentence_lower.split()
    random.shuffle(drugs_list)
    for drug in drugs_list:
        skip_drug = False
        for existing_drug in drug_mentions:
            if drug in existing_drug.drug_name or existing_drug.drug_name in drug:
                # Having overlapping drug names makes it difficult to preprocess; omit these
                skip_drug = True
                break
        if skip_drug:
            continue
        if is_sublist(drug.split(), sentence_tokens):
            # Sample one instance of the drug in this sentence, if it occurs multiple times.
            try:
                entity_occurrence = random.choice(list(re.finditer(drug, sentence_lower)))
            except:
                breakpoint()
            drug_entity = DrugEntity(drug_name=drug,
                                     span_start=entity_occurrence.start(),
                                     span_end=entity_occurrence.end())
            if (entity_occurrence.start() == 0 or sentence_lower[entity_occurrence.start()-1] == " ") and \
                (entity_occurrence.end() == len(sentence_lower) or sentence_lower[entity_occurrence.end()] == " "):
                # Want to find drug mentions that are standalone tokens, not contained in other entities
                drug_mentions.append(drug_entity)

    masked_documents = []
    if mask_one_drug_at_a_time:
        for drug in drug_mentions:
            sentence_marked = add_entity_markers(sentence, [drug])
            sentence_drugs_masked = replace_spans(sentence_marked, [drug.drug_name], ["[drug]"])
            paragraph_prefix_drugs_masked = replace_spans(paragraph_prefix, [drug.drug_name], ["[drug]"])
            paragraph_suffix_drugs_masked = replace_spans(paragraph_suffix, [drug.drug_name], ["[drug]"])
            concatenated = "".join([paragraph_prefix_drugs_masked, sentence_drugs_masked, paragraph_suffix_drugs_masked])
            masked_documents.append({"text": concatenated, "drugs": [drug.drug_name]})
    else:
        sentence_marked = add_entity_markers(sentence, drug_mentions)
        replacement_tokens = [f"[drug{idx}]" for idx in range(len(drug_mentions))]
        drug_spans = [drug.drug_name for drug in drug_mentions]
        sentence_drugs_masked = replace_spans(sentence_marked, drug_spans, replacement_tokens)
        paragraph_prefix_drugs_masked = replace_spans(paragraph_prefix, drug_spans, replacement_tokens)
        paragraph_suffix_drugs_masked = replace_spans(paragraph_suffix, drug_spans, replacement_tokens)
        concatenated = "".join([paragraph_prefix_drugs_masked, sentence_drugs_masked, paragraph_suffix_drugs_masked])
        if len(drug_spans) > 0 and len(drug_spans) <= max_arity:
            masked_documents.append({"text": concatenated, "drugs": drug_spans})
    return masked_documents

if __name__ == "__main__":
    args = parser.parse_args()
    in_file = args.in_file
    out_directory = args.out_directory

    os.makedirs(out_directory, exist_ok=True)
    train_file = open(os.path.join(out_directory, "train.jsonl"), 'wb')
    train_writer = jsonlines.Writer(train_file)
    dev_file = open(os.path.join(out_directory, "dev.jsonl"), 'wb')
    dev_writer = jsonlines.Writer(dev_file)
    test_file = open(os.path.join(out_directory, "test.jsonl"), 'wb')
    test_writer = jsonlines.Writer(test_file)

    mask_one_drug_at_a_time = args.mask_one_drug_at_a_time

    drugs = list(open(args.entities_list).read().split("\n"))
    # Normalize the drug names.
    drugs = [" ".join(drug.lower().split()) for drug in drugs]

    # Find entities in text, to then mask
    # Borrow existing preprocessing functionality wherever possible

    processed_rows = []
    sentences = set()
    for row in tqdm(csv.DictReader(open(in_file))):
        split_assignment = choose_split(args.train_fraction, args.dev_fraction)
        if row['sentence_text'] in sentences:
            # Process each sentence once
            continue
        doc_processed_rows = process_document(row['sentence_text'], row['paragraph_text'], drugs, mask_one_drug_at_a_time)
        processed_rows.extend(doc_processed_rows)
        if len(processed_rows) % 200 == 0:
            print(f"{len(processed_rows)} sentences processed")
        if split_assignment == TRAIN_SPLIT:
            train_writer.write_all(doc_processed_rows)
        elif split_assignment == DEV_SPLIT:
            dev_writer.write_all(doc_processed_rows)
        else:
            test_writer.write_all(doc_processed_rows)

    train_file.close()
    dev_file.close()
    test_file.close()
    print(f"Wrote {len(processed_rows)} rows to {out_directory}/")