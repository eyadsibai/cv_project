import numpy as np
from utils import compute_iou


def compute_precision_recall(predictions, ground_truths, iou_threshold, class_idx):
    tp = 0
    fp = 0
    fn = 0
    # Filter predictions and ground_truths by class_idx
    predictions = [pred for pred in predictions if pred[3] == class_idx]
    ground_truths = [gt for gt in ground_truths if gt[4] == class_idx]

    for gt in ground_truths:
        matched = False
        for pred in predictions:
            iou = compute_iou(pred[:4], gt[:4])
            if iou > iou_threshold:
                tp += 1
                matched = True
                predictions.remove(pred)
                break
        if not matched:
            fn += 1

    fp = len(predictions)

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def compute_ap(precisions, recalls):
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p_val = np.array([p if r >= t else 0 for p, r in zip(precisions, recalls)])
        if p_val.size > 0:
            ap += np.max(p_val)
    ap /= 11.0
    return ap