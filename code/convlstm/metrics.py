import numpy as np
import torch


def confusion_counts(prediction, target, threshold=100.0):
    pred_positive = prediction * 255.0 >= float(threshold)
    target_positive = target * 255.0 >= float(threshold)
    tp = torch.sum(pred_positive & target_positive).item()
    fp = torch.sum(pred_positive & ~target_positive).item()
    fn = torch.sum(~pred_positive & target_positive).item()
    tn = torch.sum(~pred_positive & ~target_positive).item()
    return int(tp), int(fp), int(fn), int(tn)


def scores_from_counts(tp, fp, fn):
    precision = np.nan if tp + fp == 0 else tp / (tp + fp)
    recall = np.nan if tp + fn == 0 else tp / (tp + fn)
    csi = np.nan if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return float(precision), float(recall), float(csi)

