'''
python scripts/extract_relations_from_spike.py \
    --spike-file /home/vijay/drug_interaction_synergy_sentence_large_dedup.csv \
    --model-path /home/vijay/drug-synergy-models/checkpoints_with_continued_pretraining_large_scale_2023_multiclass \
    --output-file /home/vijay/drug-synergy-models/extracted_drugs_distant_supervision_large.jsonl \
    --classifier-threshold 0.3 --batch-size 13 drug_interaction_synergy_sentence_large_updated_drugs.csv
'''

import argparse
import csv
import jsonlines
import math
import numpy as np
import sys
import time
from tqdm import tqdm
import torch
sys.path.extend(["..", "."])

from common.utils import find_mentions_in_sentence, find_sent_in_para
from modeling.model import load_model
from postprocessing import concate_tensors, extract_all_candidate_relations_for_document, find_all_relations, hash_string


parser = argparse.ArgumentParser()
parser.add_argument('--spike-file', type=str, required=True, help="Downloaded CSV file from Spike")
parser.add_argument('--model-path', type=str, required=True, help="Checkpoint directory to BertForRelation model")
parser.add_argument('--output-file', type=str, required=True, help="Output JSONLINES file to write relations to")
parser.add_argument('--drug-list', type=str, required=False, default="data/drugs.txt", help="Path to list of drugs")
parser.add_argument('--classifier-threshold', type=float, default=0.3, required=False, help="Threshold to use for classifier decisions")
parser.add_argument('--batch-size', type=int, required=False, default=32)

def load_spike_rows(spike_file):
    return csv.DictReader(open(spike_file))

def convert_spike_row_to_model_input(row, drugs_list):
    sentence_start_idx, sentence_end_idx = find_sent_in_para(row["sentence_text"], row["paragraph_text"])
    doc_id = hash_string(row["paragraph_text"])
    # Update document to insert processed sentence into unprocessed paragraph
    row["paragraph_text"] = " ".join([row["paragraph_text"][:sentence_start_idx].strip(), row["sentence_text"], row["paragraph_text"][sentence_end_idx:].strip()])
    matched_drugs, _ = find_mentions_in_sentence(row["sentence_text"], drugs_list)
    if len(matched_drugs) == 0:
        return None
    spans_object = []
    for i, drug in enumerate(matched_drugs):
        spans_object.append({"span_id": i,
                             "start": drug.span_start,
                             "end": drug.span_end,
                             "text": drug.drug_name,
                             "token_start": -1, # To be lazy, don't populate these two fields since it's not used by our model.
                             "token_end": -1})
        
    # Need to have `spans`, `paragraph`, `sentence`
    model_friendly_format = {"doc_id": doc_id, "sentence": row["sentence_text"], "paragraph": row["paragraph_text"], "spans": spans_object}
    return model_friendly_format

def divide_into_batches(model_inputs, batch_size=32):
    all_input_ids, all_token_type_ids, all_attention_masks, all_entity_idxs = model_inputs
    batches = []
    for i in range(math.ceil(all_input_ids.shape[0] / batch_size)):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        batch_input_ids = all_input_ids[start_idx:end_idx]
        batch_token_type_ids = all_token_type_ids[start_idx:end_idx]
        batch_attention_masks = all_attention_masks[start_idx:end_idx]
        batch_entity_idxs = all_entity_idxs[start_idx:end_idx]
        batches.append([batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_entity_idxs])
    return batches

def process_batch(batch, model, tokenizer, relation_labels, model_metadata, drugs_list, batch_size):
    batch_doc_ids = []
    batch_texts = []
    batch_sentences = []
    batch_candidate_relations = []
    batch_drug_idxs = []
    tensors = []
    for row in batch:
        message = convert_spike_row_to_model_input(row, drugs_list)
        if message == None:
            continue
        candidate_relations, row_tensors = extract_all_candidate_relations_for_document(message, tokenizer, model_metadata.max_seq_length, model_metadata.label2idx, model_metadata.context_window_size, model_metadata.include_paragraph_context)
        if candidate_relations is None and row_tensors is None:
            # Skip doc.
            continue

        batch_candidate_relations.extend([list(r) for r in candidate_relations])
        batch_doc_ids.extend([str(message["doc_id"]) for _ in candidate_relations])
        batch_texts.extend([message["paragraph"] for _ in candidate_relations])
        batch_sentences.extend([message["sentence"] for _ in candidate_relations])
        for relation in candidate_relations:
            relation_drug_idxs = []
            for drug_name in relation:
                matching_span_idxs = [span["span_id"] for span in message["spans"] if span["text"] == drug_name]
                assert len(matching_span_idxs) > 0
                relation_drug_idxs.append(matching_span_idxs[0])
            batch_drug_idxs.append(sorted(relation_drug_idxs))
        tensors.append(row_tensors)

    model_inputs = concate_tensors(tensors)
    input_batches = divide_into_batches(model_inputs, batch_size=batch_size)

    all_relation_probs = []
    for batch in input_batches:
        all_input_ids, all_token_type_ids, all_attention_masks, all_entity_idxs = batch
        all_input_ids = all_input_ids.cuda()
        all_token_type_ids = all_token_type_ids.cuda()
        all_attention_masks = all_attention_masks.cuda()
        all_entity_idxs = all_entity_idxs.cuda()

        model_inputs = [all_input_ids, all_token_type_ids, all_attention_masks, None, all_entity_idxs, None]
        with torch.cuda.amp.autocast():
            softmaxes = model.predict_probabilities(model_inputs)
        all_relation_probs.extend(softmaxes.detach().cpu().tolist())
        del all_input_ids
        del all_token_type_ids
        del all_attention_masks
        del all_entity_idxs
        del model_inputs
        torch.cuda.empty_cache()
    assert len(all_relation_probs) == len(batch_doc_ids), breakpoint()

    jlines = []
    for i in range(len(all_relation_probs)):
        relation_probabilities = dict(zip(relation_labels, all_relation_probs[i]))
        line = {"drug_combination": batch_candidate_relations[i],
                "relation_probabilities": relation_probabilities,
                "paragraph": batch_texts[i],
                "sentence": batch_sentences[i],
                "doc_id": batch_doc_ids[i],
                "drug_idxs": batch_drug_idxs[i]}
        jlines.append(line)
    return jlines

if __name__ == "__main__":
    args = parser.parse_args()
    drugs_list = open(args.drug_list).read().lower().split("\n")
    spike_rows = load_spike_rows(args.spike_file)
    model, tokenizer, metadata = load_model(args.model_path)
    idx2label = {idx:label for label, idx in metadata.label2idx.items()}
    relation_labels = [idx2label[idx] for idx in range(len(idx2label))]
    model = model.cuda()
    model.eval()

    MEM_BATCH_SIZE=350
    num_rows_processed = 0
    rows_batch = []

    start = time.perf_counter()
    with open(args.output_file, 'w') as outfile:
        jsonl_writer = jsonlines.Writer(outfile)
        for row in tqdm(spike_rows):
            if len(rows_batch) == MEM_BATCH_SIZE:
                outlines = process_batch(rows_batch, model, tokenizer, relation_labels, metadata, drugs_list, args.batch_size)
                jsonl_writer.write_all(outlines)
                rows_batch = []
            num_rows_processed += 1
            if num_rows_processed % 1000 == 0 or (num_rows_processed < 1000 and num_rows_processed % 100 == 0):
                now = time.perf_counter()
                print(f"{num_rows_processed} rows processed in {round(now - start, 3)} seconds")
            rows_batch.append(row)
        # Process final batch
        if len(rows_batch) > 0:
            outlines = process_batch(rows_batch, model, tokenizer, relation_labels, metadata, drugs_list, args.batch_size)
            jsonl_writer.write_all(outlines)