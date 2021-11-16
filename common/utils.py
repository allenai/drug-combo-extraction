import csv
import json
import jsonlines
import numpy as np
import os
import random
import re
import torch
from typing import List, Dict, Tuple, Any

from common.types import DrugEntity


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy of predictions against ground truth.

    Args:
        Predictions: tensor of binary predictions
        Labels: tensor of binary ground truth labels

    Returns:
        acc: Accuracy of predictions
    """
    acc = float(torch.sum(predictions == labels).item()) / len(predictions)
    return acc

def true_positives(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Helper function to compute number of true positives (which is the
    numerator for both precision and recall).

    Args:
        Predictions: tensor of binary predictions
        Labels: tensor of binary ground truth labels

    Returns:
        tps: Number of true positives
    """
    tps = float(torch.sum(torch.logical_and(predictions == 1.0, labels == 1.0)).item())
    return tps

def precision(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute precision of predictions against ground truth.

    Args:
        Predictions: tensor of binary predictions
        Labels: tensor of binary ground truth labels

    Returns:
        acc: Precision of predictions
    """
    predicted_pos = float(torch.sum(predictions))
    true_pos = true_positives(predictions, labels)
    if true_pos == 0.0:
        return 0.0
    return true_pos/predicted_pos

def recall(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute recall of predictions against ground truth.

    Args:
        Predictions: tensor of binary predictions
        Labels: tensor of binary ground truth labels

    Returns:
        acc: Recall of predictions
    """
    gt_pos = float(torch.sum(labels))
    true_pos = true_positives(predictions, labels)
    if gt_pos == 0.0:
        return 0.0
    return true_pos/gt_pos

def compute_f1(preds: torch.Tensor, labels: torch.Tensor):
    """Compute the F1 score of predictions against ground truth. Return as a dictionary including precision
    and recall, since these must be computed to calculate F1.
    Exact f1 calculation was copied almost verbatim from
    https://github.com/princeton-nlp/PURE/blob/8517005d947afedcbb2b04df9d8de18fa1ca9b04/run_relation.py#L164-L189.

    Args:
        Predictions: tensor of binary predictions
        Labels: tensor of binary ground truth labels

    Returns:
        F1: F1 score of predictions
        prec: Precision of predictions
        rec: Recall of predictions
    """
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1, 'n_correct': n_correct, 'n_pred': n_pred, 'task_ngold': n_gold}

def read_jsonl(fname: str):
    return list(jsonlines.open(fname))

def write_jsonl(data: List[Dict], fname: str):
    with open(fname, 'wb') as fp:
        writer = jsonlines.Writer(fp)
        writer.write_all(data)
    writer.close()
    print(f"Wrote {len(data)} json lines to {fname}")

def write_json(data: Dict, fname: str):
    json.dump(data, open(fname, 'w'), indent=4)
    print(f"Wrote json file to {fname}")

class ModelMetadata:
    def __init__(self,
                 model_name: str,
                 max_seq_length: int,
                 num_labels: int,
                 label2idx: Dict,
                 add_no_combination_relations: bool,
                 only_include_binary_no_comb_relations: bool,
                 include_paragraph_context: bool,
                 context_window_size: int):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.label2idx = label2idx
        self.add_no_combination_relations = add_no_combination_relations
        self.only_include_binary_no_comb_relations = only_include_binary_no_comb_relations
        self.include_paragraph_context = include_paragraph_context
        self.context_window_size = context_window_size


def save_metadata(metadata: ModelMetadata, checkpoint_directory: str):
    '''Serialize metadata about a model and the data preprocessing that it expects, to allow easy model usage at a later time.

    Args:
        metadata: ModelMetadata object containing information needed to use the model after loading a checkpoint.
        checkpoint_directory: Directory name in which to save the metadata file ($checkpoint_directory/metadata.json)
    '''
    metadata_dict = {
        "model_name": metadata.model_name,
        "max_seq_length":  metadata.max_seq_length,
        "num_labels":  metadata.num_labels,
        "label2idx":  metadata.label2idx,
        "add_no_combination_relations":  metadata.add_no_combination_relations,
        "only_include_binary_no_comb_relations":  metadata.only_include_binary_no_comb_relations,
        "include_paragraph_context":  metadata.include_paragraph_context,
        "context_window_size":  metadata.context_window_size
    }
    metadata_file = os.path.join(checkpoint_directory, "metadata.json")
    json.dump(metadata_dict, open(metadata_file, 'w'))

def load_metadata(checkpoint_directory: str) -> ModelMetadata:
    '''Given a directory containing a model checkpoint, metadata regarding the model and data preprocessing that the model expects.

    Args:
        checkpoint_directory: Path to local directory where model is serialized

    Returns:
        metadata: ModelMetadata object containing information needed to use the model after loading a checkpoint.
    '''
    metadata_file = os.path.join(checkpoint_directory, "metadata.json")
    metadata_dict = json.load(open(metadata_file))
    metadata = ModelMetadata(metadata_dict["model_name"],
                             metadata_dict["max_seq_length"],
                             metadata_dict["num_labels"],
                             metadata_dict["label2idx"],
                             metadata_dict["add_no_combination_relations"],
                             metadata_dict["only_include_binary_no_comb_relations"],
                             metadata_dict["include_paragraph_context"],
                             metadata_dict["context_window_size"])
    return metadata

def construct_row_id_idx_mapping(dataset: List[Dict]) -> Tuple[Dict, Dict]:
    '''For a list of dataset rows, which contain string-hash row IDs, map these
    into integers (for the purposes of tensorization), and return the mapping.

    Args:
        dataset: list of JSON rows, representing individual relations in our dataset (each containing a row_id field)

    Returns:
        row_id_idx_mapping: mapping from row_id strings to integer indices
        idx_row_id_mapping: reverse mapping from integer indices to row_id strings
    '''
    row_id_idx_mapping = {}
    idx_row_id_mapping = {}
    for doc in dataset:
        idx = len(row_id_idx_mapping)
        row_id_idx_mapping[doc["row_id"]] = idx
        idx_row_id_mapping[idx] = doc["row_id"]
    return row_id_idx_mapping, idx_row_id_mapping

def average_pairwise_distance(spans: List[Dict]) -> float:
    '''This function calculates the average distance between pairs of spans in a relation, which may be a useful
    bucketing attribute for error analysis.

    Args:
        spans: List of spans (each represented as a dictionary)

    Returns:
        average pairwise distance between spans in the provided list
    '''
    distances = []
    for i in range(len(spans)):
        for j in range(i+1, len(spans)):
            span_distance = spans[j]["token_start"] > spans[i]["token_end"]
            if span_distance >= 0:
                distances.append(span_distance)
            else:
                span_distance = spans[i]["token_start"] - spans[j]["token_end"]
                assert span_distance >= 0
                distances.append(span_distance)
    assert len(distances) >= 1
    return np.mean(distances)

class ErrorAnalysisAttributes:
    def __init__(self, dataset_row: Dict, full_document: Dict, prediction: int):
        self.row_id = dataset_row["row_id"]
        self.sentence = full_document["sentence"]
        self.paragraph = full_document["paragraph"]
        self.sentence_length = len(full_document["sentence"].split())
        self.paragraph_length = len(full_document["paragraph"].split())
        spans = full_document["spans"]

        self.entities = [span["text"] for span in spans]
        spans_in_relation = [spans[idx] for idx in dataset_row["drug_indices"]]

        self.num_spans_in_ground_truth_relation = len(spans_in_relation)
        self.avg_span_distance_in_ground_truth_relation  = average_pairwise_distance(spans_in_relation)
        self.ground_truth_label = dataset_row["target"]
        self.predicted_label = prediction

    def get_row(self):
        return [self.row_id, self.sentence, self.entities, self.paragraph, self.ground_truth_label, self.predicted_label, self.sentence_length, self.paragraph_length, self.num_spans_in_ground_truth_relation, self.avg_span_distance_in_ground_truth_relation]


def write_error_analysis_file(dataset: List[Dict], test_data_raw: List[Dict], test_row_ids: List[str], test_predictions: List[int], fname: str):
    '''Write out all test set rows and their predictions to a TSV file, which will let us connect easily with ExplainaBoard.

    Args:
        dataset: List of row dictionaries representing the test dataset
        test_row_ids: List of row identifiers in the test set
        test_predictions: List of integer predictions corresponding to the test rows
        fname: String file to write the TSV output
    '''
    test_data_raw = {doc["doc_id"]:doc for doc in test_data_raw}
    row_predictions = dict(zip(test_row_ids, test_predictions))

    header = [
                "Row ID",
                "Sentence",
                "Entities",
                "Paragraph",
                "True Relation Label",
                "Predicted Relation Label",
                "Sentence Length",
                "Paragraph Length",
                "Number of Entities in Ground Truth Relation",
                "Average Distance of Entities"
            ]

    with open(fname, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(header)
        for dataset_row in dataset:
            prediction = row_predictions[dataset_row["row_id"]]
            doc_id = json.loads(dataset_row["row_id"])["doc_id"]
            full_document = test_data_raw[doc_id]
            error_analysis_attributes = ErrorAnalysisAttributes(dataset_row, full_document, prediction)
            tsv_writer.writerow(error_analysis_attributes.get_row())
    print(f"Wrote error analysis file to {fname}")

def set_seed(seed):
    # set seed for all possible avenues of stochasticity
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adjust_data(gold: List[str], test: List[int]) -> List[Dict[str, Any]]:
    """Given a list of row id strings and their corresponding predicted labels, convert the prediction list
        to a list of the same format as the gold, where the only difference would be the predicted label.

    Args:
        gold: List of Dictionary object, each of which represents a row_id.
            A row ID contains a sentence hash, a list of drug indices, and a gold relation label.
        test: List of integers representing a predicted relation label

    Returns:
        fixed test lists of "row id" dictionary
    """
    fix_t = []
    for i, (g, t) in enumerate(zip(gold, test)):
        # for the test/pred we want to the copied information from the gold list,
        #   except for the relation label itself
        fix_t.append(json.loads(g))
        fix_t[i]["relation_label"] = t
    return fix_t


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
            if e["relation_label"]:
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

def find_sent_in_para(sent, para):
    para = para.replace("\u2009", " ").replace("\u00a0", " ").replace("\u202f", " ").replace("\u2003", " ").replace("\u200a", " ")
    idx = para.replace(" ", "").find(sent.replace(" ", ""))
    c = 0
    for i in range(idx):
        while para[i + c] == " ":
            c += 1
    c2 = 0
    for i in range(len(sent.replace(" ", ""))):
        while i + idx + c + c2 < len(para) and para[i + idx + c + c2] == " ":
            c2 += 1
    return idx + c, idx + c + c2 + len(sent.replace(" ", ""))

def is_sublist(list_a, list_b):
    if len(list_a) == 0:
        return True
    for i in range(len(list_b) - len(list_a)+1):
        if list_b[i] == list_a[0]:
            matched = True
            for j in range(1, len(list_a)):
                if list_a[j] != list_b[i+j]:
                    matched = False
                    break
            if matched:
                return True
    return False

def find_mentions_in_sentence(sentence, drug_list):
    drug_mentions = []
    drug_repetition_idxs = []
    sentence_lower = sentence.lower()
    sentence_tokens = sentence_lower.split()
    for drug in drug_list:
        skip_drug = False
        for existing_drug in drug_mentions:
            if drug in existing_drug.drug_name or existing_drug.drug_name in drug:
                # Having overlapping drug names makes it difficult to preprocess; omit these
                skip_drug = True
                break
        if skip_drug:
            continue
        if is_sublist(drug.split(), sentence_tokens):
            # Sample one instance of the drug in this sentence, if it occurs multiple times.
            entity_occurrences = []
            for occurrence in re.finditer(re.escape(drug), sentence_lower):
                # Want to find drug mentions that are standalone tokens, not contained in other entities
                if (occurrence.start() == 0 or sentence_lower[occurrence.start()-1] == " ") and \
                    (occurrence.end() == len(sentence_lower) or sentence_lower[occurrence.end()] == " "):
                    entity_occurrences.append(occurrence)
            if len(entity_occurrences) == 0:
                return [], []
            entity_occurrence_idx = random.choice(list(range(len(entity_occurrences))))
            entity_occurrence = entity_occurrences[entity_occurrence_idx]
            drug_entity = DrugEntity(drug_name=drug,
                                     drug_idx=len(drug_mentions),
                                     span_start=entity_occurrence.start(),
                                     span_end=entity_occurrence.end())
            drug_mentions.append(drug_entity)
            drug_repetition_idxs.append(entity_occurrence_idx)
    return drug_mentions, drug_repetition_idxs
