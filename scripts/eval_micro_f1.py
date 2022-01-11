import argparse
import json
from enum import Enum
import sklearn.metrics
from typing import List, Dict, Any, Tuple

parser = argparse.ArgumentParser()
parser.add_argument('--gold-file', type=str, required=True, default="data/unittest_gold.jsonl", help="Path to the gold file")
parser.add_argument('--pred-file', type=str, required=True, default="data/unittest_pred.jsonl", help="Path to the predictions file")


class Label(Enum):
    NO_COMB = 0
    NEG = 1
    COMB = 2
    POS = 3
    NEG_AND_COMB = 4


def get_label_pos_comb(rel):
    str_label2idx = {"true": 1, "false": 0, "NO_COMB": 0}
    int_label2idx = {1: 1, 0: 0, 2: 0}
    if type(rel['relation_label']) == str:
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
    return idx_label

def create_vectors(gold: List[Dict[str, Any]], test: List[Dict[str, Any]]) \
        -> Tuple[Dict[Tuple[str, str, int], List[Tuple[int, float]]],
                 Dict[Tuple[str, str, int], List[Tuple[int, float]]]]:
    """This function constructs the gold and predicted vectors such that each gold/prediction,
        would be mapped to a list of its aligned counterparts. this alignment is needed for later metrics.

    Args:
        gold: a list of gold dictionaries each of which stands for a relation.
            each has a doc_id to identify which doc did it came from, drug_idxs to pinpoint the drugs participating in this relation,
            and a relation_label to state the gold labels.
        test: the same as gold but having the predicted labels instead.

    Example:
        gold: [{'doc_id': 1, 'drug_idxs': [1, 2], 'relation_label': 3}, {'doc_id': 2, 'drug_idxs': [0, 1], 'relation_label': 1}]
        test: [{'doc_id': 1, 'drug_idxs': [0, 1, 2], 'relation_label': 3}, {'doc_id': 2, 'drug_idxs': [0, 1], 'relation_label': 0}]
        unify negs: False
        exact match: False
        =>
        g_out: {(1, '[1, 2]', 3): [(3, 0.666)], (2, '[0, 1]', 1): [(0, 0)]}
        t_out: {(1, '[0, 1, 2]', 3): [(3, 0.666)]}

    Returns:
        gold and test dictionaries that map from each relation to its (partial/exact) matched labels and their scores
    """
    union = {}
    for rel1 in test:
        found = False
        for rel2 in gold:
            if rel1['doc_id'] != rel2['doc_id']:
                continue
            if set(rel1['drug_idxs']) == set(rel2['drug_idxs']):
                found = True
                break
        if found:
            label_pair = (get_label_pos_comb(rel1), get_label_pos_comb(rel2))
        else:
            label_pair = (get_label_pos_comb(rel1), 0)
        union[(rel1["doc_id"], str(rel1["drug_idxs"]))] = label_pair

    for rel2 in gold:
        for rel1 in test:
            if rel1['doc_id'] != rel2['doc_id']:
                continue
            if set(rel1['drug_idxs']) == set(rel2['drug_idxs']):
                found = True
                break
        if not found:
            label_pair = (0, get_label_pos_comb(rel2))
            union[(rel2["doc_id"], str(rel2["drug_idxs"]))] = label_pair
    return union

def micro_f1(vectors):
    y_true = []
    y_pred = []
    for (t, p) in vectors.values():
        y_true.append(t)
        y_pred.append(p)
    return sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, labels=[0,1], average='micro')

def f_score(gold, test):
    union_vectors = create_vectors(gold, test)
    p, r, f, support = micro_f1(union_vectors)
    return f, p, r

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.pred_file) as f:
        pred = [json.loads(l) for l in f.readlines()]
    with open(args.gold_file) as f:
        gold = [json.loads(l) for l in f.readlines()]

    vectors = create_vectors(pred, gold)

    f_l, p_l, r_l = f_score(gold, pred)
    print(f"labeled = {f_l, p_l, r_l}")
