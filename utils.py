import json
<<<<<<< HEAD
import jsonlines
=======
import os
>>>>>>> Refactor code to make it easier to call one-off model inference
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

def save_metadata(model_name, max_seq_length, num_labels, label2idx, include_paragraph_context, checkpoint_directory):
    metadata = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "num_labels": num_labels,
        "label2idx": label2idx,
        "include_paragraph_context": include_paragraph_context
    }
    metadata_file = os.path.join(checkpoint_directory, "metadata.json")
    json.dump(metadata, open(metadata_file, 'w'))

def load_metadata(checkpoint_directory):
    metadata_file = os.path.join(checkpoint_directory, "metadata.json")
    metadata = json.load(open(metadata_file))
    return metadata["model_name"], metadata["max_seq_length"], metadata["num_labels"], metadata["label2idx"], metadata["include_paragraph_context"]
