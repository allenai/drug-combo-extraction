import csv
import numpy as np
import torch
from typing import Dict, List, Tuple

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

def f1(predictions: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    """Compute the F1 score of predictions against ground truth. Also return precision
    and recall, since these must be computed to calculate F1.

    Args:
        Predictions: tensor of binary predictions
        Labels: tensor of binary ground truth labels

    Returns:
        F1: F1 score of predictions
        prec: Precision of predictions
        rec: Recall of predictions
    """
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    if prec+rec == 0.0:
        f = 0.0
    else:
        f = 2*((prec*rec)/(prec+rec))
    return f, prec, rec

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
            if span_distance:
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
