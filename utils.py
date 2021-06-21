import torch

def accuracy(predictions, labels):
    acc = float(torch.sum(predictions == labels).item()) / len(predictions)
    return acc

def true_positives(predictions, labels):
    tps = float(torch.sum(torch.logical_and(predictions == 1.0, labels == 1.0)).item())
    return tps

def precision(predictions, labels):
    predicted_pos = float(torch.sum(predictions))
    true_pos = true_positives(predictions, labels)
    if true_pos == 0.0:
        return 0.0
    return true_pos/predicted_pos

def recall(predictions, labels):
    gt_pos = float(torch.sum(labels))
    true_pos = true_positives(predictions, labels)
    if gt_pos == 0.0:
        return 0.0
    return true_pos/gt_pos

def f1(predictions, labels):
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    if prec+rec == 0.0:
        f = 0.0
    else:
        f = 2*((prec*rec)/(prec+rec))
    return f, prec, rec