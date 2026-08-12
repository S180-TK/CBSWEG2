import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.plotting import PlotGeneration


def test_display_test_accuracies_runs_without_error():
    PlotGeneration().display_test_accuracies(0.5, 0.6, 0.65)
    plt.close('all')


def test_display_learning_curves_runs_for_mlr_and_mlp():
    plotter = PlotGeneration()
    history = [0.3, 0.5, 0.7, 0.8]

    mlr_model = {'learning_rate': 0.1, 'epochs': 4, 'batch_size': 32}
    plotter.display_learning_curves(history, 0.75, mlr_model, 'MLR')
    plt.close('all')

    mlp_model = {'hidden_size': 16, 'learning_rate': 0.1, 'epochs': 4, 'batch_size': 32}
    plotter.display_learning_curves(history, 0.75, mlp_model, 'MLP')
    plt.close('all')


def test_display_loss_curves_runs_without_error():
    history = [0.9, 0.7, 0.5, 0.3]
    model = {'learning_rate': 0.1, 'epochs': 4, 'batch_size': 32, 'validation_loss': 0.4}
    PlotGeneration().display_loss_curves(history, model, 'MLR')
    plt.close('all')


def test_plot_confusion_matrices_runs_without_error():
    cm_a = np.array([[5, 1], [2, 4]])
    cm_b = np.array([[6, 0], [1, 5]])
    PlotGeneration().plot_confusion_matrices(cm_a, cm_b, ['Class A', 'Class B'], 'Model A', 'Model B')
    plt.close('all')


def test_compare_metrics_bar_runs_without_error():
    metrics_df = pd.DataFrame({
        'Accuracy': [0.5, 0.6],
        'Balanced Accuracy': [0.4, 0.5],
        'Precision (macro)': [0.4, 0.5],
        'Recall (macro)': [0.4, 0.5],
        'F1-score (macro)': [0.4, 0.5],
        'PR-AUC (macro)': [0.4, 0.5],
    }, index=['Logistic Regression', 'MLP (Neural Network)'])
    PlotGeneration().compare_metrics_bar(metrics_df)
    plt.close('all')
