import torch

def accuracy(predictions, labels):
    acc = float(torch.sum(predictions == labels).item()) / len(predictions)
    return acc

def precision(predictions, labels):
    predicted_positives = float(torch.sum(predictions))
    true_positives = float(torch.sum(predictions == labels).item())
    return true_positives/predicted_positives

def recall(predictions, labels):
    gt_positives = float(torch.sum(labels))
    true_positives = float(torch.sum(predictions == labels).item())
    return true_positives/gt_positives

def f1(predictions, labels):
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    f1 = 2*((prec*rec)/(prec+rec))
    return f1, prec, rec