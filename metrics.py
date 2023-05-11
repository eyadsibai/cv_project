import numpy as np
from utils import compute_iou


def compute_precision_recall(predictions, ground_truths, iou_threshold):
    tp = 0
    fp = 0
    fn = 0
    for gt in ground_truths:
        matched = False
        for pred in predictions:
            iou = compute_iou(pred[:4], gt)
            if iou > iou_threshold:
                tp += 1
                matched = True
                # predictions.remove(pred)
                break
            
        if not matched:
            fn += 1
    
    for pred in predictions:
        matched = False
        for gt in ground_truths:
            iou = compute_iou(pred[:4], gt)
            if iou > iou_threshold:
                matched = True
                # ground_truths.remove(pred)
                break
            
        if not matched:
            fp += 1
        
    # fp = len(predictions)

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def compute_ap(precision, recall, predictions):
    # ap = 0
    # for t in np.arange(0, 1.1, 0.1):
    #     p_val = np.array([p if r >= t else 0 for p, r in zip(precisions, recalls)])
    #     if p_val.size > 0:
    #         ap += np.max(p_val)
    # ap /= 11.0
    # return ap
    
     # Compute PR curve for class i
    pr_curves = []
    pr_curve = np.zeros((2, len(predictions)))
    pr_curve[0] = np.array(recall)
    pr_curve[1] = np.array(precision)
    sorted_indices = np.argsort(np.array(predictions)[:, 4])[::-1]
    pr_curve[0] = pr_curve[0][sorted_indices]
    pr_curve[1] = pr_curve[1][sorted_indices]
    pr_curves.append(pr_curve)