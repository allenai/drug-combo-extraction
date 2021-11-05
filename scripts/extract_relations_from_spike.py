'''
python scripts/extract_relations_from_spike.py \
    --spike-file /Users/vijay/Downloads/distant_supervision_small.csv \
    --model-path /tmp/ \
    --output-file /tmp/output_rows.csv \
    --classifier-threshold 0.3
'''

import argparse
import csv
import sys
sys.path.extend(["..", "."])
from tqdm import tqdm

from common.utils import find_mentions_in_sentence, find_sent_in_para
from modeling.model import load_model
from postprocessing import concate_tensors, extract_all_candidate_relations_for_document, find_all_relations, hash_string


parser = argparse.ArgumentParser()
parser.add_argument('--spike-file', type=str, required=True, help="Downloaded CSV file from Spike")
parser.add_argument('--model-path', type=str, required=True, help="Checkpoint directory to BertForRelation model")
parser.add_argument('--output-file', type=str, required=True, help="Output CSV file to write relations to")
parser.add_argument('--drug-list', type=str, required=False, default="data/drugs.txt", help="Path to list of drugs")
parser.add_argument('--classifier-threshold', type=float, default=0.3, required=False, help="Threshold to use for classifier decisions")

def load_spike_rows(spike_file):
    return csv.DictReader(open(spike_file))

def convert_spike_row_to_model_input(row, drugs_list):
    sentence_start_idx, sentence_end_idx = find_sent_in_para(row["sentence_text"], row["paragraph_text"])
    doc_id = hash_string(row["sentence_text"])
    # Update document to insert processed sentence into unprocessed paragraph
    row["paragraph_text"] = " ".join([row["paragraph_text"][:sentence_start_idx].strip(), row["sentence_text"], row["paragraph_text"][sentence_end_idx:].strip()])
    matched_drugs, _ = find_mentions_in_sentence(row["sentence_text"], drugs_list)
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

if __name__ == "__main__":
    args = parser.parse_args()
    drugs = open(args.drug_list).read().lower().split("\n")
    spike_rows = load_spike_rows(args.spike_file)
    model, tokenizer, metadata = load_model(args.model_path)
    
    rows_per_document = []
    tensors = []
    for row in tqdm(spike_rows):
        message = convert_spike_row_to_model_input(row, drugs)
        num_rows, row_tensors = extract_all_candidate_relations_for_document(message, tokenizer, metadata.max_seq_length, metadata.label2idx, metadata.context_window_size, metadata.include_paragraph_context)
        if num_rows is None and row_tensors is None:
            # Skip doc.
            continue
        rows_per_document.append(num_rows)
        tensors.append(row_tensors)

    model_inputs = concate_tensors(tensors)