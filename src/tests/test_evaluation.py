import numpy as np
import pytest

from models.evaluation import summarize_metrics


def test_summarize_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_true_onehot = np.eye(3)[y_true]
    y_probs = y_true_onehot.astype(float)

    result = summarize_metrics(y_true, y_pred, y_probs, y_true_onehot, 'Perfect Model')

    assert result['Model'] == 'Perfect Model'
    assert result['Accuracy'] == pytest.approx(1.0)
    assert result['Balanced Accuracy'] == pytest.approx(1.0)
    assert result['Precision (macro)'] == pytest.approx(1.0)
    assert result['Recall (macro)'] == pytest.approx(1.0)
    assert result['F1-score (macro)'] == pytest.approx(1.0)
    assert result['PR-AUC (macro)'] == pytest.approx(1.0)


def test_summarize_metrics_known_imperfect_case():
    # 2 classes, 4 samples: one misclassification (an actual class-1 sample predicted as class 0)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1])
    y_true_onehot = np.eye(2)[y_true]
    y_probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.2, 0.8],
    ])

    result = summarize_metrics(y_true, y_pred, y_probs, y_true_onehot, 'Test Model')

    # class 0: precision 2/3, recall 2/2=1.0 ; class 1: precision 1/1=1.0, recall 1/2=0.5
    assert result['Accuracy'] == pytest.approx(3 / 4)
    assert result['Precision (macro)'] == pytest.approx((2 / 3 + 1.0) / 2)
    assert result['Recall (macro)'] == pytest.approx((1.0 + 0.5) / 2)
