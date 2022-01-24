'''
python scripts/add_ddi_drugs.py \
  --training-data /home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/train.jsonl \
  --current-drug-list /home/vijay/updated_drugs.txt \
  --new-drug-list /home/vijay/updated_drugs_with_ddi_drugs.txt
'''

import argparse
import json
import jsonlines
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--training-data', type=str, required=False, default="/home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/train.jsonl", help="Training set to use to add new entities")
parser.add_argument('--current-drug-list', type=str, required=False, default="/home/vijay/updated_drugs.txt")
parser.add_argument('--new-drug-list', type=str, required=False, default="/home/vijay/updated_drugs_with_ddi_drugs.txt")

if __name__ == "__main__":
    args = parser.parse_args()

    drug_list = [l.strip() for l in open(args.current_drug_list).readlines()]
    processed_drug_list = []
    for d in drug_list:
        if d.endswith("(human)"):
            d = d[:-len("(human)")]
        processed_drug_list.append(d)
    drug_set = set(processed_drug_list)

    dataset_entities = []
    dataset_relations = []
    dataset = list(jsonlines.open(args.training_data))
    for row in dataset:
      row_entities = {span["span_id"] : span["text"].lower() for span in row["spans"]}
      for rel in row["rels"]:
        rel_spans = [row_entities[span_idx] for span_idx in rel["spans"]]
        dataset_relations.append(tuple(sorted(rel_spans)))
        dataset_entities.extend(rel_spans)

    drug_set = list(drug_set.union(dataset_entities))
    outfile = open(args.new_drug_list, 'w')
    outfile.write("\n".join(drug_set))
    outfile.close()
