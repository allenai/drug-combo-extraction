import json
from enum import Enum
from collections import defaultdict


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
    g_out = defaultdict(list)
    t_out = defaultdict(list)
    matched = set()
    for rel1 in gold:
        found = False
        for k, rel2 in enumerate(test):
            if rel1['doc_id'] != rel2['doc_id']:
                continue
            rel2_spans = set(rel2['drug_idxs'])
            spans_intersecting = 0
            # count matching entity-spans
            for span in rel1['drug_idxs']:
                for span2 in rel2_spans:
                    if span == span2:
                        rel2_spans.remove(span2)
                        spans_intersecting += 1
                        break
            # we have at least partial matching
            if ((spans_intersecting >= 2) and (not exact_match)) or ((spans_intersecting / len(set(rel1['drug_idxs'] + list(rel2_spans)))) == 1):
                score = spans_intersecting / len(set(rel1['drug_idxs'] + list(rel2_spans)))
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1, unify_negs))].append((get_label(rel2, unify_negs), score, rel2["drug_idxs"]))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2, unify_negs))].append((get_label(rel1, unify_negs), score, rel1["drug_idxs"]))
                found = True
                matched.add(k)
        # if a gold positive not found by test, add a false negative pair
        if not found:
            g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1, unify_negs))].append((Label.NO_COMB.value, 0, None))
    # no we iterate of the remaining relations in the test, and add the false positives
    for k, rel2 in enumerate(test):
        if k not in matched:
            t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2, unify_negs))].append((Label.NO_COMB.value, 0, None))
    return g_out, t_out


def f_from_p_r(gs, ts, labeled=False):
    def get_max_sum_score(v):
        interesting = 0
        score = 0
        for (_, _, label), matched in v.items():
            if label != Label.NO_COMB.value:
                interesting += 1
                score += max([s if ((not labeled and other != Label.NO_COMB.value) or (other == label)) else 0 for other, s, _ in matched])
        return score / interesting
    p = get_max_sum_score(ts)
    r = get_max_sum_score(gs)
    return (2 * p * r) / (p + r), p, r


def adjust_data(gold, test):
    fix_g = []
    fix_t = []
    for i, (g, t) in enumerate(zip(gold, test)):
        fix_g.append(json.loads(g))
        fix_t.append(json.loads(g))
        fix_t[i]["relation_label"] = t
    return fix_g, fix_t


def filter_overloaded_predictions(fixed_test):
    def do_filtering(d):
        out = d[0]
        for j, e in d:
            if e["relation_label"] != Label.NO_COMB.value:
                out = (j, e)
                break
        send_to_further = []
        for j, e in d:
            if len(set(out[1]["drug_idxs"]).intersection(set(e["drug_idxs"]))) == 0:
                send_to_further.append((j, e))
        return [out] + (do_filtering(send_to_further) if send_to_further else [])

    final_test = []
    sorted_test = sorted(enumerate(fixed_test), key=lambda x: (x[1]["doc_id"], len(x[1]["drug_idxs"]), str(x[1]["drug_idxs"])), reverse=True)
    doc = []
    for i, (original_idx, example) in enumerate(sorted_test):
        doc.append((original_idx, example))
        if (i + 1 == len(sorted_test)) or (sorted_test[i + 1][1]["doc_id"] != example["doc_id"]):
            final_test.extend(do_filtering(doc))
            doc = []
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
    return f, f_labeled
