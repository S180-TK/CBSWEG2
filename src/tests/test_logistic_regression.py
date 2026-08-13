import numpy as np
import pytest

from models.logistic_regression import MultinomialLogisticRegression


@pytest.fixture
def model():
    np.random.seed(0)
    return MultinomialLogisticRegression(num_features=2, num_classes=3)


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


def test_compute_gradients_shapes_and_zero_case(model):
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    probs = y_true.astype(float)  # perfect predictions -> zero error
    weight_grad, bias_grad = model.compute_gradients(X, y_true, probs)
    assert weight_grad.shape == (2, 3)
    assert bias_grad.shape == (3,)
    np.testing.assert_allclose(weight_grad, np.zeros((2, 3)), atol=1e-9)
    np.testing.assert_allclose(bias_grad, np.zeros(3), atol=1e-9)


def test_compute_accuracy_matches_known_labels(model):
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    y_true = np.array([0, 2, 0])
    assert model.compute_accuracy(probs, y_true) == pytest.approx(2 / 3)


def test_compute_accuracy_accepts_one_hot_targets(model):
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    y_true = np.array([[1, 0, 0], [0, 0, 1]])
    assert model.compute_accuracy(probs, y_true) == pytest.approx(1.0)


def test_get_set_weights_round_trip(model):
    weights, bias = model.get_weight_and_bias()
    new_weights = weights + 1
    new_bias = bias + 1
    model.set_weights((new_weights, new_bias))
    result_weights, result_bias = model.get_weight_and_bias()
    np.testing.assert_allclose(result_weights, new_weights)
    np.testing.assert_allclose(result_bias, new_bias)


def test_reset_weights_reinitializes_with_correct_shape(model):
    before, _ = model.get_weight_and_bias()
    model.reset_weights()
    after, bias_after = model.get_weight_and_bias()
    assert after.shape == before.shape
    assert bias_after.shape == (3,)
    assert not np.allclose(before, after)


def test_update_weights_applies_exact_gradient_step(model):
    weights_before, bias_before = model.get_weight_and_bias()
    model.update_weights(
        np.ones_like(weights_before), np.ones_like(bias_before), learning_rate=0.5
    )
    weights_after, bias_after = model.get_weight_and_bias()
    np.testing.assert_allclose(weights_after, weights_before - 0.5)
    np.testing.assert_allclose(bias_after, bias_before - 0.5)


def test_validation_reports_metrics_without_changing_weights(model):
    X = np.array([[1.0, -1.0], [0.5, 0.5]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    weights_before, bias_before = model.get_weight_and_bias()

    loss, accuracy = model.validation(X, y)

    # Validation must never train — early stopping and the reported test
    # metrics both call this method, so a weight update here would leak.
    weights_after, bias_after = model.get_weight_and_bias()
    np.testing.assert_array_equal(weights_after, weights_before)
    np.testing.assert_array_equal(bias_after, bias_before)

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
    model = MultinomialLogisticRegression(num_features=2, num_classes=3)
    loss_before = model.compute_loss(model.forward(X), y)

    for _ in range(200):
        model.training(X, y, learning_rate=0.1)

    loss_after = model.compute_loss(model.forward(X), y)
    assert loss_after < loss_before
