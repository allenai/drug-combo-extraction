import csv
import json
import jsonlines
import numpy as np
import torch
from typing import List, Dict, Tuple

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
    with jsonlines.Writer(open(fname, 'wb')) as writer:
        writer.write_all(data)
    print(f"Wrote {len(data)} json lines to {fname}")

def write_json(data: Dict, fname: str):
    json.dump(data, open(fname, 'w'), indent=4)
    print(f"Wrote json file to {fname}")

def construct_row_id_idx_mapping(dataset: List[Dict]) -> Tuple[Dict, Dict]:
    '''For a list of dataset rows, which contain string-hash row IDs, map these
    into integers (for the purposes of tensorization), and return the mapping.
    '''
    row_id_idx_mapping = {}
    idx_row_id_mapping = {}
    for doc in dataset:
        idx = len(row_id_idx_mapping)
        row_id_idx_mapping[doc["row_id"]] = idx
        idx_row_id_mapping[idx] = doc["row_id"]
    return row_id_idx_mapping, idx_row_id_mapping

def average_pairwise_distance(spans):
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
        return [self.sentence, self.entities, self.paragraph, self.ground_truth_label, self.predicted_label, self.sentence_length, self.paragraph_length, self.num_spans_in_ground_truth_relation, self.avg_span_distance_in_ground_truth_relation]

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
            doc_id = dataset_row["row_id"].split("_rels_")[0]
            full_document = test_data_raw[doc_id]
            error_analysis_attributes = ErrorAnalysisAttributes(dataset_row, full_document, prediction)
            tsv_writer.writerow(error_analysis_attributes.get_row())
    print(f"Wrote error analysis file to {out_file}")
