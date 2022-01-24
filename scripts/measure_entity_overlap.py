'''
python scripts/measure_entity_overlap.py \
  /home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/train.jsonl \
  /home/vijay/checkpoints_large_scale_relation_prediction_lr_5e_5_min_rel_freq_25_epochs_15/metadata.json
'''

import json
import jsonlines
import sys

if __name__ == "__main__":
    dataset = list(jsonlines.open(sys.argv[1]))
    metadata = json.load(open(sys.argv[2]))

    entities = []
    relations = []
    for relation, index in metadata["label2idx"].items():
      relation_tuple = tuple(sorted([v.lower() for v in json.loads(relation)]))
      if index != -1:
        relations.append(relation_tuple)
        entities.extend(list(relation_tuple))

    relations = set(relations)
    entities = set(entities)

    dataset_entities = []
    dataset_relations = []
    for row in dataset:
      row_entities = {span["span_id"] : span["text"].lower() for span in row["spans"]}
      for rel in row["rels"]:
        rel_spans = [row_entities[span_idx] for span_idx in rel["spans"]]
        dataset_relations.append(tuple(sorted(rel_spans)))
        dataset_entities.extend(rel_spans)

    relations_covered = [r for r in dataset_relations if r in relations]
    relation_coverage = float(len(relations_covered)) / len(dataset_relations)
    entities_covered = [r for r in dataset_entities if r in entities]
    entity_coverage = float(len(entities_covered)) / len(dataset_entities)
    print(f"Relation Coverage: {round(relation_coverage * 100, 2)}%")
    print(f"Entity Coverage: {round(entity_coverage * 100, 2)}%")

