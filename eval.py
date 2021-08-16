import argparse
import json
from enum import Enum
from collections import defaultdict
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


labels = {3: Label.POS.value, 1: Label.COMB.value, 2: Label.NEG.value, 0: Label.NO_COMB.value}
labels2 = {3: Label.POS.value, 1: Label.NEG_AND_COMB.value, 2: Label.NEG_AND_COMB.value, 0: Label.NO_COMB.value}
labels3 = {1: Label.POS.value, 0: Label.NO_COMB.value}


def get_label(rel, unify_negs):
    return labels[rel['relation_label']] if not unify_negs else labels3[rel['relation_label']]


def create_vectors(gold, test, unify_negs, exact_match):
    """This function constructs the gold and predicted vectors such that each gold/prediction,
        would be mapped to a list of its aligned counterparts. this alignment is needed for later metrics.
    """
    g_out = defaultdict(list)
    t_out = defaultdict(list)
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
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1, unify_negs))].append((get_label(rel2, unify_negs), score))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2, unify_negs))].append((get_label(rel1, unify_negs), score))
                found = True
                matched.add(k)
        # if a gold positive not found by test, add a false negative pair
        if not found:
            g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1, unify_negs))].append((Label.NO_COMB.value, 0))
    # now we iterate on the remaining relations in the test, and add the false positives
    for k, rel2 in enumerate(test):
        if k not in matched:
            t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2, unify_negs))].append((Label.NO_COMB.value, 0))
    return g_out, t_out


def f_from_p_r(gs, ts, labeled=False):
    def get_max_sum_score(v):
        interesting = 0
        score = 0
        for (_, _, label), matched in v.items():
            if label != Label.NO_COMB.value:
                interesting += 1
                score += max([s if ((not labeled and other != Label.NO_COMB.value) or (other == label)) else 0 for other, s in matched])
        return score / interesting
    p = get_max_sum_score(ts)
    r = get_max_sum_score(gs)
    return (2 * p * r) / (p + r), p, r


def adjust_data(gold: List[str], test: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Given a list of row id strings and their corresponding predicted labels, convert the gold row id to its underlying dictionary,
        and the prediction list to a list of the same format, where the only difference would be the gold and predicted label.

    Args:
        gold: List of Dictionary oblect, each of which represents a row_id.
            A row ID contains a sentence hash, a list of drug indices, and a gold relation label.
        test: List of integers representing a predicted relation label

    Returns:
        fixed gold and test lists of "row id" dictionary
    """
    fix_g = []
    fix_t = []
    for i, (g, t) in enumerate(zip(gold, test)):
        fix_g.append(json.loads(g))

        # for the test/pred we want to the copied information from the gold list,
        #   except for the relation label itself
        fix_t.append(json.loads(g))
        fix_t[i]["relation_label"] = t
    return fix_g, fix_t


def filter_overloaded_predictions(preds):
    def do_filtering(d):
        # our filtering algorithm:
        #   1. we assume each sentence gets predictions for each subset of drugs in the sentence
        #   2. we assume these are too many and probably conflicting predictions, so they need to be filtered
        #   3. we use a very simple (greedy) heuristic in which we look for the biggest (by drug-count) combination,
        #       that has a non NO_COMB prediction, and we take it.
        #   4. we try to get as large a coverage (on the drugs) as possible while maintaining
        #       a minimalistic list of predictions as possible, so we do this repeatedly on the remaining drugs
        out = d[0]
        for j, e in d:
            if e["relation_label"] != Label.NO_COMB.value:
                out = (j, e)
                break
        send_to_further = []
        for j, e in d:
            # store all non intersecting predictions with the chosen one, so we can repeat the filtering process on them
            if len(set(out[1]["drug_idxs"]).intersection(set(e["drug_idxs"]))) == 0:
                send_to_further.append((j, e))
        return [out] + (do_filtering(send_to_further) if send_to_further else [])

    # we sort here so it would be easier to group by sentence,
    #   and to have the high-drug-count examples first for the filtering process
    sorted_test = sorted(enumerate(preds), key=lambda x: (x[1]["doc_id"], len(x[1]["drug_idxs"]), str(x[1]["drug_idxs"])), reverse=True)

    # aggregate predictions by the sentence and filter each prediction group
    final_test = []
    doc = []
    for i, (original_idx, example) in enumerate(sorted_test):
        doc.append((original_idx, example))
        # reached the last one in the list, or last one for this sentence
        if (i + 1 == len(sorted_test)) or (sorted_test[i + 1][1]["doc_id"] != example["doc_id"]):
            final_test.extend(do_filtering(doc))
            doc = []
    # reorder the filtered list according to original indices, and get rid of the these indices
    return [x[1] for x in sorted(final_test, key=lambda x: x[0])]


def f_score_our_model(gold, test, unify_negs=False, exact_match=False):
    fixed_gold, fixed_test = adjust_data(gold, test)
    fixed_test = filter_overloaded_predictions(fixed_test)
    return f_score(fixed_gold, fixed_test, unify_negs, exact_match)


def f_score(gold, test, unify_negs=False, exact_match=False):
    gs, ts = create_vectors(gold, test, unify_negs, exact_match)
    f, p, r = f_from_p_r(gs, ts)
    f_labeled, p_l, r_l = f_from_p_r(gs, ts, labeled=True)
    print(f"F1/P/R score: unlabeled = {f, p, r}, labeled = {f_labeled, p_l, r_l}")
    return f, p, r, f_labeled, p_l, r_l


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.pred_file) as f:
        pred = [json.loads(l) for l in f.readlines()]
    with open(args.gold_file) as f:
        gold = [json.loads(l) for l in f.readlines()]
    ret = f_score(gold, pred)
    # TODO (Aryeh): make this a "real" unit test at some point
    if args.pred_file == "data/unittest_pred.jsonl":
        assert ret == (0.5950540958268934, 0.6481481481481481, 0.55, 0.3760886777513856, 0.4629629629629629, 0.31666666666666665)
