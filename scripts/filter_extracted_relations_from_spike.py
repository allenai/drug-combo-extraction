'''
python scripts/filter_extracted_relations_from_spike.py \
    --extracted-relations-raw /home/vijay/drug-synergy-models/extracted_drugs_larger_query_pubmedbert_three_class.jsonl \
    --extracted-relations-processed /home/vijay/drug-synergy-models/knowledge_base_pubmedbert_three_class.jsonl \
    --model-metadata /home/vijay/checkpoints_pubmedbert_cpt_2021_three_class/metadata.json
'''

import argparse
from collections import Counter
import json
import jsonlines
import sys

sys.path.extend(['.', '..'])
from common.utils import filter_overloaded_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--extracted-relations-raw', type=str, required=True, help="Jsonlines file containing raw relation probabilities")
parser.add_argument('--extracted-relations-processed', type=str, required=True, help="File containing filtered, discrete relations")
parser.add_argument('--model-metadata', type=str, required=True, help="Model metadata JSON file generated from model training")
parser.add_argument('--pos-threshold', type=float, default=0.999, help="Threshold to classify a relation as POS")
parser.add_argument('--neg-threshold', type=float, default=0.99, help="Threshold to classify a relation as COMB")

if __name__ == "__main__":
    args = parser.parse_args()
    label2idx = json.load(open(args.model_metadata))["label2idx"]
    thresholded_relations = []
    for raw_extraction in jsonlines.open(args.extracted_relations_raw):
        # These extractions contain label probabilities, but have not been thresholded.
        if raw_extraction["relation_probabilities"]["POS"] > args.pos_threshold:
            raw_extraction["relation_label"] = "POS"
        elif raw_extraction["relation_probabilities"]["NEG"] > args.neg_threshold:
            raw_extraction["relation_label"] = "COMB"
        else:
            raw_extraction["relation_label"] = "NO_COMB"
        thresholded_relations.append(raw_extraction)

    postprocessed_relations = filter_overloaded_predictions(thresholded_relations)
    unique_relations = {"POS": set(), "COMB": set(), "NO_COMB": set()}
    for r in postprocessed_relations:
        unique_relations[r["relation_label"]].add(tuple(r["drug_combination"]))
    unique_relation_counts = {k: len(unique_relations[k]) for k, v in unique_relations.items()}

    print(f"Extracted relation counts: {unique_relation_counts}.")

    sorting_order = {"POS": 0, "NEG": 1, "COMB": 2, "NO_COMB": 3}

    postprocessed_relations = sorted(postprocessed_relations, key=lambda k: (sorting_order[k["relation_label"]], k["drug_combination"]))

    with open(args.extracted_relations_processed, 'w') as outfile:
        outwriter = jsonlines.Writer(outfile)
        for row in postprocessed_relations:
            rounded_probabilities = {k: round(v, 6) for k, v in row["relation_probabilities"].items()}
            simplified = {"drug_combination": row["drug_combination"], "predicted_label": row["relation_label"], "relation_probabilities": rounded_probabilities, "evidence_sentence": row["sentence"], "evidence_paragraph": row["paragraph"]}
            outwriter.write(simplified)
