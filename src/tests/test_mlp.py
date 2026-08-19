import numpy as np
import pytest

from models.mlp import MLP


@pytest.fixture
def model():
    np.random.seed(0)
    return MLP(num_features=2, hidden_size=4, num_classes=3)


def test_relu_zeros_out_negative_values(model):
    z = np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]])
    np.testing.assert_allclose(model.relu(z), [[0.0, 0.0, 0.0, 1.5, 3.0]])


def test_relu_derivative_matches_step_function(model):
    z = np.array([[-1.0, 0.0, 2.0]])
    np.testing.assert_allclose(model.relu_derivative(z), [[0.0, 0.0, 1.0]])


def test_softmax_rows_sum_to_one(model):
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = model.softmax(logits.copy())
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-9)


def test_forward_returns_valid_probability_distribution(model):
    X = np.array([[1.0, -1.0], [0.5, 0.5]])
    probs = model.forward(X)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-9)
    assert np.all(probs > 0)


def test_compute_loss_matches_hand_computed_cross_entropy(model):
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    expected = -np.mean([np.log(0.7), np.log(0.8)])
    assert model.compute_loss(probs, y_true) == pytest.approx(expected)


def test_compute_accuracy_matches_known_labels(model):
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    y_true = np.array([0, 2, 0])
    assert model.compute_accuracy(probs, y_true) == pytest.approx(2 / 3)


def test_get_set_weights_round_trip(model):
    weights = model.get_weight_and_bias()
    new_weights = tuple(w + 1 for w in weights)
    model.set_weights(new_weights)
    result = model.get_weight_and_bias()
    for expected, actual in zip(new_weights, result):
        np.testing.assert_allclose(actual, expected)


def test_reset_weights_reinitializes_with_correct_shapes(model):
    before = model.get_weight_and_bias()
    model.reset_weights()
    after = model.get_weight_and_bias()
    for b, a in zip(before, after):
        assert b.shape == a.shape
    assert not np.allclose(before[0], after[0])


def test_update_weights_applies_exact_gradient_step(model):
    before = model.get_weight_and_bias()
    gradients = tuple(np.ones_like(array) for array in before)
    model.update_weights(*gradients, learning_rate=0.5)
    after = model.get_weight_and_bias()
    for expected, actual in zip(before, after):
        np.testing.assert_allclose(actual, expected - 0.5)


def test_validation_reports_metrics_without_changing_weights(model):
    X = np.array([[1.0, -1.0], [0.5, 0.5]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    before = model.get_weight_and_bias()

    loss, accuracy = model.validation(X, y)

    # Validation must never train — early stopping and the reported test
    # metrics both call this method, so a weight update here would leak.
    for expected, actual in zip(before, model.get_weight_and_bias()):
        np.testing.assert_array_equal(actual, expected)

    probs = model.forward(X)
    assert loss == pytest.approx(model.compute_loss(probs, y))
    assert accuracy == pytest.approx(model.compute_accuracy(probs, y))


def test_testing_matches_validation_on_the_same_data(model):
    X = np.array([[1.0, -1.0], [0.5, 0.5]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    test_loss, test_accuracy = model.testing(X, y)
    val_loss, val_accuracy = model.validation(X, y)
    assert test_loss == pytest.approx(val_loss)
    assert test_accuracy == pytest.approx(val_accuracy)


def test_training_reduces_loss_on_separable_toy_data():
    np.random.seed(0)
    X = np.array([
        [5.0, 0.0], [4.5, 0.5],
        [0.0, 5.0], [0.5, 4.5],
        [-5.0, -5.0], [-4.5, -4.5],
    ])
    y = np.array([
        [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1],
    ])
    model = MLP(num_features=2, hidden_size=8, num_classes=3)
    loss_before = model.compute_loss(model.forward(X), y)

    for _ in range(200):
        model.training(X, y, learning_rate=0.1)

    loss_after = model.compute_loss(model.forward(X), y)
    assert loss_after < loss_before
