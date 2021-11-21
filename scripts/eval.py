import argparse
import json
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Any, Tuple

parser = argparse.ArgumentParser()
parser.add_argument('--gold-file', type=str, required=True, default="data/unittest_gold.jsonl", help="Path to the gold file")
parser.add_argument('--pred-file', type=str, required=True, default="data/unittest_pred.jsonl", help="Path to the predictions file")
parser.add_argument('--exact-match', action='store_true', help="whether to preform an exact match (bo partials)")


class Label(Enum):
    NO_COMB = 0
    NEG = 1
    COMB = 2
    POS = 3
    NEG_AND_COMB = 4


def get_label_pos_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 0, "COMB": 0, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 0, 0: 0}
    if type(rel['relation_label']) == str:
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
    return idx_label

def get_label_any_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 1, "COMB": 1, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 1, 0: 0}
    if type(rel['relation_label']) == str:
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
    return idx_label

def create_vectors(gold: List[Dict[str, Any]], test: List[Dict[str, Any]], exact_match: bool, any_comb: bool) \
        -> Tuple[Dict[Tuple[str, str, int], List[Tuple[int, float]]],
                 Dict[Tuple[str, str, int], List[Tuple[int, float]]]]:
    """This function constructs the gold and predicted vectors such that each gold/prediction,
        would be mapped to a list of its aligned counterparts. this alignment is needed for later metrics.

    Args:
        gold: a list of gold dictionaries each of which stands for a relation.
            each has a doc_id to identify which doc did it came from, drug_idxs to pinpoint the drugs participating in this relation,
            and a relation_label to state the gold labels.
        test: the same as gold but having the predicted labels instead.
        exact_match: if True, restricts the matching criteria to be have the same spans in both relations.
            default is False, which gives the partial matching behavior in which we require at least two spans in common

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
    g_out = defaultdict(list)
    t_out = defaultdict(list)
    if any_comb:
        get_label = get_label_any_comb
    else:
        get_label = get_label_pos_comb

    matched = set()
    for rel1 in gold:
        found = False
        for k, rel2 in enumerate(test):
            if rel1['doc_id'] != rel2['doc_id']:
                continue
            # count matching entity-spans
            spans_intersecting = len(set(rel1['drug_idxs']).intersection(set(rel2['drug_idxs'])))
            score = spans_intersecting / len(set(rel1['drug_idxs'] + rel2['drug_idxs']))
            # if we have partial matching (when allowed) or exact matching (when required) add the aligned relations
            if ((spans_intersecting >= 2) and (not exact_match)) or (score == 1):
                # we use as mapping the "row id" (sentence hash, drug indices, and the label). and we map a list
                #   of aligned relations (+ scores) of the other vector
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), score))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), score))
                found = True
                matched.add(k)
        # if a gold positive not found by test, add a false negative pair
        if not found:
            g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((Label.NO_COMB.value, 0))
    # now we iterate on the remaining relations in the test, and add the false positives
    for k, rel2 in enumerate(test):
        if k not in matched:
            t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((Label.NO_COMB.value, 0))
    return g_out, t_out


def get_max_sum_score(v, labeled):
    interesting = 0
    score = 0
    for (_, _, label), matched in v:
        if label != Label.NO_COMB.value:
            interesting += 1
            score += max([s if ((not labeled and other != Label.NO_COMB.value) or (other == label)) else 0 for other, s in matched])
    return score / interesting

def f_from_p_r(gs, ts, labeled=False):
    p = get_max_sum_score(ts.items(), labeled)
    r = get_max_sum_score(gs.items(), labeled)
    return (2 * p * r) / (p + r), p, r


def f_score(gold, test, exact_match=False, any_comb=False):
    gs, ts = create_vectors(gold, test, exact_match, any_comb=any_comb)
    f, p, r = f_from_p_r(gs, ts)
    return f, p, r


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.pred_file) as f:
        pred = [json.loads(l) for l in f.readlines()]
    with open(args.gold_file) as f:
        gold = [json.loads(l) for l in f.readlines()]
    f, p, r = f_score(gold, pred, exact_match=args.exact_match, any_comb=True)
    f_l, p_l, r_l = f_score(gold, pred, exact_match=args.exact_match, any_comb=False)
    print(f"F1/P/R score: unlabeled = {f, p, r}, labeled = {f_l, p_l, r_l}")

    # TODO (Aryeh): make this a "real" unit test at some point
    if args.pred_file == "data/unittest_pred.jsonl":
        partial_f, partial_p, partial_r = f_score(gold, pred, exact_match=not args.exact_match, any_comb=True)
        partial_f_l, partial_p_l, partial_r_l = f_score(gold, pred, exact_match=not args.exact_match, any_comb=False)
        ret = (f, p, r, f_l, p_l, r_l)
        ret2 = (partial_f, partial_p, partial_r, partial_f_l, partial_p_l, partial_r_l)
        scores = {False: (0.5950540958268934, 0.6481481481481481, 0.55, 0.3760886777513856, 0.4629629629629629, 0.31666666666666665),
                  True: (0.3157894736842105, 0.3333333333333333, 0.3, 0.2105263157894737, 0.2222222222222222, 0.2)}
        assert ret == scores[args.exact_match]
        assert ret2 == scores[not args.exact_match]
