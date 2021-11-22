# Usage
# python scripts/high_confidence_precision.py \
#        --checkpoint-path /home/vijay/checkpoints_pubmedbert_cpt_2021_three_class \
#        --test-file data/final_test_set.jsonl \
#        --batch-size 70 \
#        --seed 2021

import argparse
import json
import jsonlines
import os
import numpy as np
import pytorch_lightning as pl
import sys

sys.path.extend(['.', '..'])
from common.utils import construct_row_id_idx_mapping, set_seed, write_error_analysis_file, write_jsonl, adjust_data, filter_overloaded_predictions
from modeling.model import RelationExtractor, load_model
from preprocessing.data_loader import  DrugSynergyDataModule
from preprocessing.preprocess import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str, required=False, default="checkpoints", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--test-file', type=str, required=False, default="data/dev_set_error_analysis.jsonl")
parser.add_argument('--batch-size', type=int, default=32, help="Batch size for testing (larger batch -> faster evaluation)")
parser.add_argument('--error-analysis-file', type=str, required=False, help="Output file containing error analysis information", default="test_output.tsv")
parser.add_argument('--seed', type=int, required=False, default=2021)
parser.add_argument('--pos-threshold', type=float, default=0.999, help="Threshold to apply for extracting POS relations")
parser.add_argument('--comb-threshold', type=float, default=0.999, help="Threshold to apply for extracting COMB relations")

def generate_high_confidence_predictions(pred_proba, label2idx, pos_threshold, comb_threshold):
    predictions = []
    for i in range(len(pred_proba)):
        pred_idx = np.argmax(pred_proba[i])
        if pred_idx == label2idx["POS"] and pred_proba[i][pred_idx] > pos_threshold:
            predictions.append(label2idx["POS"])
        elif pred_idx == label2idx["COMB"] and pred_proba[i][pred_idx] > comb_threshold:
            predictions.append(label2idx["COMB"])
        else:
            predictions.append(label2idx["NO_COMB"])
    return predictions

def compute_high_confidence_precision(test_data, fixed_high_confidence, label2idx):
    true_positives = 0
    false_positives = 0

    true_labels = {}
    for i in range(len(test_data)):
        test_row = test_data[i]
        row_meta = json.loads(test_row["row_id"])
        true_label = row_meta["relation_label"]
        true_labels[(row_meta["doc_id"], tuple(sorted(row_meta["drug_idxs"])))] = true_label

    for high_confidence_pred in fixed_high_confidence:
        true_label = true_labels.get((high_confidence_pred["doc_id"], tuple(sorted(high_confidence_pred["drug_idxs"]))), label2idx["NO_COMB"])
        if high_confidence_pred["relation_label"] in [label2idx["POS"], label2idx["COMB"]]:
            if true_label == high_confidence_pred["relation_label"]:
                true_positives += 1
            else:
                false_positives += 1

    num_high_confidence_predictions = true_positives + false_positives
    precision = float(true_positives) / num_high_confidence_predictions
    return precision, num_high_confidence_predictions

if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)
    model, tokenizer, metadata = load_model(args.checkpoint_path)

    test_data_raw = list(jsonlines.open(args.test_file))
    # TODO(Vijay): add `add_no_combination_relations`, `only_include_binary_no_comb_relations`, `include_paragraph_context`,
    # `context_window_size` to the model's metadata
    test_data = create_dataset(test_data_raw,
                               label2idx=metadata.label2idx,
                               add_no_combination_relations=metadata.add_no_combination_relations,
                               only_include_binary_no_comb_relations=metadata.only_include_binary_no_comb_relations,
                               include_paragraph_context=metadata.include_paragraph_context,
                               context_window_size=metadata.context_window_size,
                               produce_all_subsets=True)
    row_id_idx_mapping, idx_row_id_mapping = construct_row_id_idx_mapping(test_data)
    dm = DrugSynergyDataModule(None,
                               test_data,
                               tokenizer,
                               metadata.label2idx,
                               row_id_idx_mapping,
                               train_batch_size=args.batch_size,
                               dev_batch_size=args.batch_size,
                               test_batch_size=args.batch_size,
                               max_seq_length=metadata.max_seq_length,
                               balance_training_batch_labels=False)
    dm.setup()

    system = RelationExtractor(model, 0, tokenizer=tokenizer)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        resume_from_checkpoint=os.path.join(args.checkpoint_path, "model.chkpt")
    )
    trainer.test(system, datamodule=dm)

    test_predictions = system.test_predictions
    test_pred_probas = system.test_pred_probas

    high_confidence_predictions = generate_high_confidence_predictions(test_pred_probas, metadata.label2idx, args.pos_threshold, args.comb_threshold)
    test_row_ids = [idx_row_id_mapping[row_idx] for row_idx in system.test_row_idxs]
    fixed_high_confidence = filter_overloaded_predictions(adjust_data(test_row_ids, high_confidence_predictions))

    high_confidence_positive_precision, num_high_confidence_positive_predictions = compute_high_confidence_precision(test_data, fixed_high_confidence, metadata.label2idx)
    print(f"High confidence precision is {high_confidence_positive_precision}, based on {num_high_confidence_positive_predictions} predictions with probability over {args.pos_threshold} for POS and {args.comb_threshold} for COMB.")