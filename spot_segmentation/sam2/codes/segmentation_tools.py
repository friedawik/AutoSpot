import numpy as np
from IPython import embed

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def pixel_level_metrics(pred_mask, gt_mask):

    # Ensure that the masks are numpy arrays and flattened
    predicted_mask = np.asarray(pred_mask).flatten()
    ground_truth_mask = np.asarray(gt_mask).flatten()

    # Calculate TP, FP, FN, TN, and Total
    TP = np.sum((predicted_mask == 1) & (ground_truth_mask == 1))
    FP = np.sum((predicted_mask == 1) & (ground_truth_mask == 0))
    FN = np.sum((predicted_mask == 0) & (ground_truth_mask == 1))
    TN = np.sum((predicted_mask == 0) & (ground_truth_mask == 0))
    Total = TP + FP + FN + TN
    # embed()
    # Calculate precision, recall, F1 score, IoU, and mean accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    #accuracy = (TP + TN) / Total if Total > 0 else 0

        # Balanced Accuracy
    TPR = recall  # True Positive Rate
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate
    balanced_accuracy = (TPR + TNR) / 2
    
    return precision, recall, f1_score, iou, balanced_accuracy