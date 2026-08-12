import numpy as np
from sklearn.preprocessing import StandardScaler

from models.logistic_regression import MultinomialLogisticRegression
from models.training import run_kfold_valid, train_and_evaluate_final_model


def make_toy_data():
    X = np.array([
        [5.0, 0.0], [4.8, 0.2], [4.6, -0.1], [4.9, 0.1],
        [0.0, 5.0], [0.2, 4.8], [-0.1, 4.6], [0.1, 4.9],
        [-5.0, -5.0], [-4.8, -4.9], [-4.9, -5.1], [-5.1, -4.8],
    ])
    y_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y = np.eye(3)[y_labels]
    return X, y


def test_run_kfold_valid_returns_per_fold_results():
    np.random.seed(0)
    X, y = make_toy_data()
    model = MultinomialLogisticRegression(num_features=2, num_classes=3)
    scaler = StandardScaler()

    fold_accuracies, fold_losses = run_kfold_valid(
        model, X, y, learning_rate=0.1, max_epochs=5, k=2,
        batch_size=4, patience_limit=2, cols_to_scale=[0, 1], scaler=scaler,
    )

    assert len(fold_accuracies) == 2
    assert len(fold_losses) == 2
    assert all(0.0 <= acc <= 1.0 for acc in fold_accuracies)


def test_train_and_evaluate_final_model_returns_expected_results():
    np.random.seed(0)
    X, y = make_toy_data()

    # Every split covers all three classes so the model is never asked
    # to validate/test on a class it never saw during training.
    train_idx = [0, 1, 4, 5, 8, 9]
    val_idx = [2, 6, 10]
    test_idx = [3, 7, 11]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = MultinomialLogisticRegression(num_features=2, num_classes=3)
    results, train_acc_history, train_loss_history = train_and_evaluate_final_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        lr=0.1, max_epochs=30, batch_size=4, patience_limit=10,
    )

    expected_keys = {
        "learning_rate", "epochs", "batch_size", "patience_limit",
        "training_accuracy", "validation_loss", "validation_accuracy",
        "testing_loss", "testing_accuracy",
    }
    assert expected_keys.issubset(results.keys())
    assert 0.0 <= results["testing_accuracy"] <= 1.0
    assert len(train_acc_history) <= 30
    assert len(train_acc_history) == len(train_loss_history)
