# This script performs a bucketing analysis to identify performance differences of our strongest baseline model
# along different splits of data. We use a multi-bootstrap procedure (https://arxiv.org/abs/2106.16163) to estimate
# the metric distributions on each split.

'''
python produce_gold_jsonl.py /home/vijay/drug-synergy-models/data/final_test_set.jsonl /home/vijay/drug-synergy-models/data/final_test_rows.jsonl

python scripts/bucketing_analysis.py --pred-files \
/home/vijay/drug-synergy-models/checkpoints_pubmedbert_cpt_2021_three_class/outputs/predictions.jsonl \
/home/vijay/drug-synergy-models/checkpoints_pubmedbert_cpt_2022_three_class/outputs/predictions.jsonl \
/home/vijay/drug-synergy-models/checkpoints_pubmedbert_cpt_2023_three_class/outputs/predictions.jsonl \
/home/vijay/drug-synergy-models/checkpoints_pubmedbert_cpt_2024_three_class/outputs/predictions.jsonl \
--gold-file /home/vijay/drug-synergy-models/data/final_test_rows.jsonl \
--bucket-type context_required
python scripts/bucketing_analysis.py --pred-files \
/home/vijay/checkpoints_pubmedbert_cpt_2021_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2022_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2023_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2024_three_class/outputs/predictions.jsonl \
--gold-file /home/vijay/drug-synergy-models/data/final_test_rows.jsonl \
--bucket-type context_required --exact-match

python scripts/bucketing_analysis.py --pred-files \
/home/vijay/checkpoints_pubmedbert_cpt_2021_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2022_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2023_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2024_three_class/outputs/predictions.jsonl \
--gold-file /home/vijay/drug-synergy-models/data/final_test_rows.jsonl \
--bucket-type arity # --exact-match

python scripts/bucketing_analysis.py --pred-files \
/home/vijay/checkpoints_pubmedbert_cpt_2021_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2022_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2023_three_class/outputs/predictions.jsonl \
/home/vijay/checkpoints_pubmedbert_cpt_2024_three_class/outputs/predictions.jsonl \
--gold-file /home/vijay/drug-synergy-models/data/final_test_rows.jsonl \
--bucket-type relations_seen_in_training # --exact-match
'''

import argparse
from collections import defaultdict
import jsonlines
import random
import numpy as np
from typing import List, Dict, Any, Tuple

import sys
sys.path.extend(["..", "."])
from eval import create_vectors, get_max_sum_score
from preprocessing.preprocess import powerset

parser = argparse.ArgumentParser()
parser.add_argument("--pred-files", help="Relation prediction files from a system for different seeds", nargs='+', type=str)
parser.add_argument('--gold-file', type=str, required=True, default="data/unittest_gold.jsonl", help="Path to the gold file")
parser.add_argument('--exact-match', action='store_true', help="whether to preform an exact match (or partial match)")
parser.add_argument('--raw-test-data', type=str, required=False, default="data/final_test_set.jsonl")
parser.add_argument('--raw-train-data', type=str, required=False, default="data/final_train_set.jsonl")
parser.add_argument('--bucket-type', type=str, choices=["context_required", "arity", "relations_seen_in_training"])
parser.add_argument('--unlabeled', action='store_true', help="whether to preform an exact match (bo partials)")

def construct_filter_map(raw_test_data, raw_train_data, bucket_type):
    relation_mapping = {}
    if bucket_type == "context_required":
        for doc in raw_test_data:
            context_required = []
            for rel in doc["rels"]:
                context_required.append(rel.get("is_context_needed", False))
            if len(context_required) == 0 or set(context_required) == {False}:
                relation_mapping[(doc["doc_id"], "*")] = False
            elif set(context_required) == {True}:
                relation_mapping[(doc["doc_id"], "*")] = True
    elif bucket_type in ["arity", "relations_seen_in_training"]:
        if bucket_type == "relations_seen_in_training":
            training_relations = set()
            for doc in raw_train_data:
                for rel in doc["rels"]:
                    rel_drugs = [doc["spans"][entity_idx]["text"].lower() for entity_idx in rel["spans"]]
                    rel_drugs = tuple(sorted(rel_drugs))
                    training_relations.add(rel_drugs)
        for doc in raw_test_data:
            rel_powerset = powerset([span["span_id"] for span in doc["spans"]])
            for rel in rel_powerset:
                rel = list(rel)
                if bucket_type == "arity":
                    # split relations on whether the arity is 2 or greater than 2
                    relation_mapping[(doc["doc_id"], str(rel))] = len(rel) > 2
                elif bucket_type == "relations_seen_in_training":
                    # split relations on whether the relation was observed as is in training
                    rel_drugs = [doc["spans"][entity_idx]["text"].lower() for entity_idx in rel]
                    rel_drugs = tuple(sorted(rel_drugs))
                    relation_mapping[(doc["doc_id"], str(rel))] = rel_drugs in training_relations
    return relation_mapping

def split_data(all_golds, all_preds, filter_map):
    all_golds_a = []
    all_golds_b = []
    all_preds_a = []
    all_preds_b = []
    assert len(all_golds) == len(all_preds)
    for seed in range(len(all_golds)):
        single_pred = all_preds[seed]

        single_gold = all_golds[seed]
        single_gold_split_a = {}
        single_gold_split_b = {}

        for k, v in single_gold.items():
            (docid, drug_idxs_str, _) = k
            in_filter = filter_map.get((docid, drug_idxs_str), None)
            in_wildcard = filter_map.get((docid, "*"), None)
            if in_filter is False or in_wildcard is False:
                single_gold_split_a[k] = v
            elif in_filter is True or in_wildcard is True:
                single_gold_split_b[k] = v

        single_pred_split_a = {}
        single_pred_split_b = {}
        for k, v in single_pred.items():
            (docid, drug_idxs_str, _) = k
            in_filter = filter_map.get((docid, drug_idxs_str), None)
            in_wildcard = filter_map.get((docid, "*"), None)
            if in_filter is False or in_wildcard is False:
                single_pred_split_a[k] = v
            elif in_filter is True or in_wildcard is True:
                single_pred_split_b[k] = v

        all_preds_a.append(list(single_pred_split_a.items()))
        all_preds_b.append(list(single_pred_split_b.items()))
        all_golds_a.append(list(single_gold_split_a.items()))
        all_golds_b.append(list(single_gold_split_b.items()))
    return (all_preds_a, all_preds_b), (all_golds_a, all_golds_b)

def doc_construct_multibootstrap(rng, multiseed_samples, doc_ids, labeled_score):
    samples = []
    seed_idxs = list(range(len(multiseed_samples)))
    multiseed_docids = [row[0][0] for row in multiseed_samples[0]]
    for docid in doc_ids:
        seed_idx = rng.choice(seed_idxs)
        # Each row in multiseed_samples consists of ((docid, drug_idxs, true label/predicted label), [(predicted label/true label, score)])
        doc_samples = [row for row in multiseed_samples[seed_idx] if row[0][0] == docid]
        samples.extend(doc_samples)
    return get_max_sum_score(samples, labeled=labeled_score)

def doc_bootstrap_comparison(multiseeds_a, multiseeds_b, doc_samples_a, doc_samples_b, nboot=1000, labeled_score=True):
    '''
    Adapted from
    https://openreview.net/pdf?id=K0E_F0gFDgA
    '''
    thetas_a = np.zeros(nboot)
    thetas_b = np.zeros(nboot)
    assert len(doc_samples_a) == len(doc_samples_b)
    for boot_ix in range(len(doc_samples_a)):
        theta_a = doc_construct_multibootstrap(rng, multiseeds_a, doc_samples_a[boot_ix], labeled_score)
        thetas_a[boot_ix] = theta_a
        theta_b = doc_construct_multibootstrap(rng, multiseeds_b, doc_samples_b[boot_ix], labeled_score)
        thetas_b[boot_ix] = theta_b
    return thetas_a, thetas_b

def sample_random_multibootstrap(rng, multiseed_samples, num_samples, labeled_score):
    samples = []
    seed_idxs = list(range(len(multiseed_samples)))
    for _ in range(num_samples):
        seed_idx = rng.choice(seed_idxs)
        # Each row in multiseed_samples consists of ((docid, drug_idxs, true label/predicted label), [(predicted label/true label, score)])
        sample = rng.choice(multiseed_samples[seed_idx])
        samples.append(sample)
    return get_max_sum_score(samples, labeled=labeled_score)

def random_bootstrap_comparison(rng, multiseeds_a, multiseeds_b, nboot=1000, labeled_score=True):
    '''
    Adapted from
    https://openreview.net/pdf?id=K0E_F0gFDgA
    '''
    thetas_a = np.zeros(nboot)
    thetas_b = np.zeros(nboot)
    num_samples_a = min([len(multiseed) for multiseed in multiseeds_a])
    num_samples_b = min([len(multiseed) for multiseed in multiseeds_b])
    for boot_ix in range(nboot):
        theta_a = sample_random_multibootstrap(rng, multiseeds_a, num_samples_a, labeled_score)
        thetas_a[boot_ix] = theta_a
        theta_b = sample_random_multibootstrap(rng, multiseeds_b, num_samples_b, labeled_score)
        thetas_b[boot_ix] = theta_b
    return thetas_a, thetas_b

def subsample_doc_ids(docids, rng, nboot=1000):
    sampled_doc_sets = []
    for _ in range(nboot):
        sampled_doc_sets.append(rng.sample(docids, k=len(docids)))
    return sampled_doc_sets

def compute_f1(precisions, recalls):
    f1 = (2 * precisions * recalls) / (precisions + recalls)
    return f1

def summarize_parameter_differences(bootstrap_samples_a, bootstrap_samples_b, metric_name):
    num_better = len([i for i in range(len(bootstrap_samples_a)) if bootstrap_samples_b[i] > bootstrap_samples_a[i]])
    print(f"{metric_name}: Split A has mean {round(np.mean(bootstrap_samples_a), 4)} and std {round(np.std(bootstrap_samples_a), 4)}. Split B has mean {round(np.mean(bootstrap_samples_b), 4)} and std {round(np.std(bootstrap_samples_b), 4)}. Split B is better {num_better} out of {len(bootstrap_samples_a)} times.")

if __name__ == "__main__":
    args = parser.parse_args()
    rng = random.Random(2021)
    all_preds = []
    for p in args.pred_files:
        all_preds.append(list(jsonlines.open(p)))
    gold = list(jsonlines.open(args.gold_file))
    raw_test_data = list(jsonlines.open(args.raw_test_data))
    raw_train_data = list(jsonlines.open(args.raw_train_data))

    filter_map = construct_filter_map(raw_test_data, raw_train_data, args.bucket_type)
    all_paired_golds = []
    all_paired_preds = []
    for pred in all_preds:
        gs, ts = create_vectors(gold, pred, args.exact_match, any_comb=args.unlabeled)
        all_paired_golds.append(gs)
        all_paired_preds.append(ts)
    preds_split, gold_split = split_data(all_paired_golds, all_paired_preds, filter_map)
    print(f"{len(gold_split[0][0])} labeled samples in Split A, {len(gold_split[1][0])} in Split B.")
    print(f"{len(preds_split[0][0])} predicted samples in Split A, {len(preds_split[1][0])} in Split B.")

    if args.bucket_type in ["context_required"]:
        docids_a = [docid for ((docid, relation_filter), keep_filter) in filter_map.items() if keep_filter is False]
        docids_b = [docid for ((docid, relation_filter), keep_filter) in filter_map.items() if keep_filter is True]

        a_doc_samples = subsample_doc_ids(docids_a, rng)
        b_doc_samples = subsample_doc_ids(docids_b, rng)

        precision_estimates_a, precision_estimates_b = doc_bootstrap_comparison(gold_split[0], gold_split[1], a_doc_samples, b_doc_samples, labeled_score=not args.unlabeled)
        recall_estimates_a, recall_estimates_b = doc_bootstrap_comparison(preds_split[0], preds_split[1], a_doc_samples, b_doc_samples, labeled_score=not args.unlabeled)
        f1s_a = compute_f1(precision_estimates_a, recall_estimates_a)
        f1s_b = compute_f1(precision_estimates_b, recall_estimates_b)

        summarize_parameter_differences(precision_estimates_a, precision_estimates_b, f"Precision, splitting on {args.bucket_type}")
        summarize_parameter_differences(recall_estimates_a, recall_estimates_b, f"Recall, splitting on {args.bucket_type}")
        summarize_parameter_differences(f1s_a, f1s_b, f"F1, splitting on {args.bucket_type}")
    elif args.bucket_type in ["arity", "relations_seen_in_training"]:
        precision_estimates_a, precision_estimates_b = random_bootstrap_comparison(rng, gold_split[0], gold_split[1], labeled_score=not args.unlabeled)
        recall_estimates_a, recall_estimates_b = random_bootstrap_comparison(rng, preds_split[0], preds_split[1], labeled_score=not args.unlabeled)
        summarize_parameter_differences(precision_estimates_a, precision_estimates_b, f"Precision, splitting on {args.bucket_type}")
        summarize_parameter_differences(recall_estimates_a, recall_estimates_b, f"Recall, splitting on {args.bucket_type}")