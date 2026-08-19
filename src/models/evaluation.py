import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def summarize_metrics(y_true, y_pred, y_probs, y_true_onehot, model_name):
    return {
        'Model': model_name,
        'Accuracy': np.mean(y_true == y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision (macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall (macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1-score (macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'PR-AUC (macro)': average_precision_score(y_true_onehot, y_probs, average='macro'),
    }
