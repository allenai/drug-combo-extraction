'''
python preprocess_pretraining_data.py --in-file /Users/vijay/Downloads/distant_supervision_large.csv \
     --out-directory /Users/vijay/Downloads/pretraining_data_small \
     --mask-one-drug-at-a-time \
     --entities-list drugs.txt
'''

import argparse
from collections import Counter
import csv
import hashlib
import json
import jsonlines
import os
import re
import random
import sys
from tqdm import tqdm
random.seed(2021)

sys.path.extend(["..", "."])
from preprocessing.preprocess import add_entity_markers, DrugEntity
from common.utils import find_sent_in_para, is_sublist

TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

parser = argparse.ArgumentParser()
parser.add_argument('--in-file', type=str, required=True)
parser.add_argument('--supervised-data', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--entities-list', type=str, required=True)
parser.add_argument('--mask-one-drug-at-a-time', action="store_true")
parser.add_argument('--dont-mask-paragraph-context', action="store_true")
parser.add_argument('--train-fraction', type=float, default=0.8)
parser.add_argument('--dev-fraction', type=float, default=0.1)

def replace_spans(text, spans, replacement_spans, replace_one_instance_of_span_in_sentence=False, spans_to_replace=None):
    for i, (span, replacement) in enumerate(zip(spans, replacement_spans)):
        span_occurrences = []
        for occurrence in re.finditer(re.escape(span), text.lower()):
            if (occurrence.start() == 0 or text[occurrence.start() - 1] == " ") and \
                (occurrence.end() == len(text) or text[occurrence.end()] == " "):
                # Do not replace partial token matches.
                span_occurrences.append(occurrence)
        if replace_one_instance_of_span_in_sentence:
            span_occurrences = [span_occurrences[spans_to_replace[i]]]
        span_occurrences = sorted(span_occurrences, key=lambda x: x.start(), reverse=True)
        for i in range(len(span_occurrences)):
            if i + 1 <= len(span_occurrences) - 1:
                assert span_occurrences[i].start() > span_occurrences[i+1].end()
            # Replace each mention with the replacement string, back to front.
            text = text[:span_occurrences[i].start()] + replacement + text[span_occurrences[i].end():]
    return text

def choose_split(train_fraction, dev_fraction):
    random_value = random.random()
    if random_value <= train_fraction:
        return TRAIN_SPLIT
    elif random_value <= train_fraction + dev_fraction:
        return DEV_SPLIT
    else:
        return TEST_SPLIT

def hash_string(string):
    return hashlib.md5((string).encode()).hexdigest()

def update_drug_entity_indices(drug_entity_object, updated_sentence, drug_token, drug_occurrence_idx):
    entity_occurrences = list(re.finditer(re.escape(drug_token), updated_sentence))
    drug_entity_object.drug_name = drug_token
    drug_entity_object.span_start = entity_occurrences[drug_occurrence_idx].start()
    drug_entity_object.span_end = entity_occurrences[drug_occurrence_idx].end()

def process_document(sentence, paragraph, article_link, drugs_list, mask_one_drug_at_a_time=True, max_arity=10):
    sentence_start_idx, sentence_end_idx = find_sent_in_para(sentence, paragraph)
    paragraph_prefix = paragraph[:sentence_start_idx]
    paragraph_suffix = paragraph[sentence_end_idx:]

    drug_mentions = []
    drug_repetition_idxs = []
    doc_hash = hash_string(sentence)
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
            entity_occurrences = []
            for occurrence in re.finditer(re.escape(drug), sentence_lower):
                # Want to find drug mentions that are standalone tokens, not contained in other entities
                if (occurrence.start() == 0 or sentence_lower[occurrence.start()-1] == " ") and \
                    (occurrence.end() == len(sentence_lower) or sentence_lower[occurrence.end()] == " "):
                    entity_occurrences.append(occurrence)
            entity_occurrence_idx = random.choice(list(range(len(entity_occurrences))))
            entity_occurrence = entity_occurrences[entity_occurrence_idx]
            drug_entity = DrugEntity(drug_name=drug,
                                     drug_idx=len(drug_mentions),
                                     span_start=entity_occurrence.start(),
                                     span_end=entity_occurrence.end())
            drug_mentions.append(drug_entity)
            drug_repetition_idxs.append(entity_occurrence_idx)

    masked_documents = []
    relation_counts = Counter()
    if len(drug_mentions) == 0:
        # No valid drug mentions in this sentence.
        return masked_documents, relation_counts

    zipped_sort = sorted(zip(drug_mentions, drug_repetition_idxs), key=lambda x: x[0].drug_name)
    drug_mentions, drug_repetition_idxs = zip(*zipped_sort)
    drug_spans = [drug.drug_name for drug in drug_mentions]
    if mask_one_drug_at_a_time:
        # In this setting, we generate N training instances from the document, where N is the number of drugs mentioned.
        if len(set(drug_spans)) >= 2:
            for drug_idx, drug in enumerate(drug_mentions):
                entity_occurrence_idx = drug_repetition_idxs[drug_idx]
                sentence_marked = add_entity_markers(sentence, [drug])
                drug_spans = [drug.drug_name]
                sentence_drugs_masked = replace_spans(sentence_marked, drug_spans, ["[drug]"], replace_one_instance_of_span_in_sentence=args.dont_mask_paragraph_context, spans_to_replace=[entity_occurrence_idx] if args.dont_mask_paragraph_context else None)
                update_drug_entity_indices(drug, sentence_drugs_masked, "[drug]", drug_occurrence_idx=0 if args.dont_mask_paragraph_context else entity_occurrence_idx)

                if not args.dont_mask_paragraph_context:
                    paragraph_prefix = replace_spans(paragraph_prefix, drug_spans, ["[drug]"])
                    paragraph_suffix = replace_spans(paragraph_suffix, drug_spans, ["[drug]"])
                concatenated = "".join([paragraph_prefix, sentence_drugs_masked, paragraph_suffix])
                spans = [{"span_id": 0, "text": drug.drug_name, "start": drug.span_start, "end": drug.span_end, "token_start": -1, "token_end": -1}]
                rels = [{"class": drug_spans, "spans": [0]}]
                masked_documents.append({"doc_id": doc_hash,
                                         "sentence": sentence_drugs_masked,
                                         "paragraph": concatenated,
                                         "drugs": drug_spans,
                                         "spans": spans,
                                         "rels": rels,
                                         "source": article_link})
                relation_counts[tuple(drug_spans)] += 1
    else:
        sentence_marked = add_entity_markers(sentence, drug_mentions)
        replacement_tokens = [f"[drug{idx}]" for idx in range(len(drug_mentions))]
        drug_spans = [drug.drug_name for drug in drug_mentions]
        sentence_drugs_masked = replace_spans(sentence_marked, drug_spans, replacement_tokens, replace_one_instance_of_span_in_sentence=args.dont_mask_paragraph_context, spans_to_replace=drug_repetition_idxs if args.dont_mask_paragraph_context else None)
        assert sentence_marked != sentence_drugs_masked, breakpoint()
        for drug_idx, drug in enumerate(drug_mentions):
            # Update drug entity
            try:
                update_drug_entity_indices(drug, sentence_drugs_masked, replacement_tokens[drug_idx], drug_occurrence_idx=0 if args.dont_mask_paragraph_context else drug_repetition_idxs[drug_idx])
            except:
                breakpoint()
        if not args.dont_mask_paragraph_context:
            paragraph_prefix = replace_spans(paragraph_prefix, drug_spans, replacement_tokens)
            paragraph_suffix = replace_spans(paragraph_suffix, drug_spans, replacement_tokens)
        concatenated = "".join([paragraph_prefix, sentence_drugs_masked, paragraph_suffix])
        # Here we store the true drug names under "class" just to maintain consistency in data loading with the other
        # dataset.
        spans = []
        for drug_idx, drug in enumerate(drug_mentions):
            spans.append({"span_id": drug_idx, "text": drug.drug_name, "start": drug.span_start, "end": drug.span_end, "token_start": -1, "token_end": -1})
        rels = [{"class": drug_spans, "spans": [span["span_id"] for span in spans]}]
        if len(set(drug_spans)) >= 2 and len(drug_spans) <= max_arity:
            masked_documents.append({"doc_id": doc_hash,
                                     "sentence": sentence_drugs_masked,
                                     "paragraph": concatenated,
                                     "drugs": drug_spans,
                                     "spans": spans,
                                     "rels": rels,
                                     "source": article_link})
            relation_counts[tuple(drug_spans)] += 1
    return masked_documents, relation_counts

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
    train_dev_file = open(os.path.join(out_directory, "train_dev.jsonl"), 'wb')
    train_dev_writer = jsonlines.Writer(train_dev_file)

    mask_one_drug_at_a_time = args.mask_one_drug_at_a_time

    drugs = list(open(args.entities_list).read().split("\n"))
    # Normalize the drug names.
    drugs = set([" ".join(drug.lower().split()) for drug in drugs])

    # Add drugs found in annotated data but not in Aryeh's list
    train_set = "/Users/vijay/Documents/code/drug-synergy-models/data/train_set.jsonl"
    test_set = "/Users/vijay/Documents/code/drug-synergy-models/data/test_set.jsonl"

    key_drugs = set()
    key_relations = set()
    for dataset in [train_set, test_set]:
        for row in jsonlines.open(dataset):
            relation_drugs = sorted([span['text'].lower() for span in row['spans']])
            if len(set(relation_drugs)) < 2:
                continue
            processed_rows, relation_counts = process_document(row['sentence'], row['paragraph'], row['source'], relation_drugs, mask_one_drug_at_a_time)
            train_writer.write_all(processed_rows)
            train_dev_writer.write_all(processed_rows)
            key_drugs.update(relation_drugs)
            key_relations.add(tuple(relation_drugs))

    drugs.update(key_drugs)
    drugs = list(drugs)

    with open(os.path.join(out_directory, "updated_drugs.txt"), 'w') as f:
        f.write("\n".join(drugs))

    # Find entities in text, to then mask
    # Borrow existing preprocessing functionality wherever possible

    processed_rows = []
    aggregated_relation_counts = Counter()
    sources = set()
    sentences = set()
    paragraphs = set()

    for row in tqdm(csv.DictReader(open(in_file))):
        split_assignment = choose_split(args.train_fraction, args.dev_fraction)
        if row['article_link'] in sources or row["sentence_text"] in sentences or row["paragraph_text"] in paragraphs:
            # Process each document once
            continue
        sources.add(row['article_link'])
        sentences.add(row['sentence_text'])
        paragraphs.add(row["paragraph_text"])
        doc_processed_rows, relation_counts = process_document(row['sentence_text'], row['paragraph_text'], row['article_link'], drugs, mask_one_drug_at_a_time)
        aggregated_relation_counts.update(relation_counts)
        processed_rows.extend(doc_processed_rows)
        if len(processed_rows) % 1000 == 0:
            print(f"{len(processed_rows)} sentences processed")
        if split_assignment == TRAIN_SPLIT:
            train_writer.write_all(doc_processed_rows)
            train_dev_writer.write_all(doc_processed_rows)
        elif split_assignment == DEV_SPLIT:
            dev_writer.write_all(doc_processed_rows)
            train_dev_writer.write_all(doc_processed_rows)
        else:
            test_writer.write_all(doc_processed_rows)

    train_file.close()
    dev_file.close()
    train_dev_writer.close()
    test_file.close()

    # Make the list of counts JSON-serializable.
    aggregated_relation_counts = {json.dumps(list(k)): v for k, v in aggregated_relation_counts.items()}
    json.dump(aggregated_relation_counts, open(os.path.join(out_directory, "relation_counts.json"), 'w'))
    json.dump(list(key_drugs), open(os.path.join(out_directory, "key_drugs.json"), 'w'))
    json.dump([json.dumps(list(k)) for k in key_relations], open(os.path.join(out_directory, "key_relations.json"), 'w'))
    print(f"Wrote {len(processed_rows)} rows to {out_directory}")