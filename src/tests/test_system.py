"""End-to-end system tests.

Unlike the unit tests, which feed small hand-built inputs to one function at a
time, each test here starts from the real `RTA_EDSA_2007-2016.csv` and pushes it
through an entire pipeline, checking the data at every stage boundary. The output
of each stage becomes the input of the next, so these tests fail when modules that
each work in isolation no longer fit together.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.clustering import (
    cluster_centroids,
    cluster_composition,
    evaluate_candidate_clusters,
    fit_final_clusters,
    prepare_modelling_features,
    summarize_clusters,
)
from models.data_processing import (
    COLLISION_TYPES,
    DataPipeline,
    UNKNOWN_COLLISION_TYPE,
    VALID_SEVERITIES,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)
from models.eda import collision_frequency_table, severity_distribution
from models.evaluation import summarize_metrics
from models.inference import cluster_collision_chi_square
from models.logistic_regression import MultinomialLogisticRegression
from models.training import train_and_evaluate_final_model

pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_SHAPE = (22072, 26)
CLEANED_ROWS = 14749


@pytest.fixture(scope="module")
def raw_df():
    """Load the real accident CSV once and share it across the system tests."""
    pipeline = DataPipeline()
    csv_path = pipeline.find_dataset_path(start=REPO_ROOT)
    assert csv_path.exists(), f"System tests need the real dataset at {csv_path}"
    return pipeline.load_dataset(csv_path)


# EDSA-ST001
def test_cbdatsi_phase2_pipeline_end_to_end(raw_df):
    """Real CSV -> cleaning -> EDA -> features -> clustering -> chi-square."""
    dp = DataPipeline()

    # Stage 1 — the raw dataset is the documented shape.
    assert raw_df.shape == RAW_SHAPE

    # Stage 2 — cleaning reduces the dataset exactly as the report describes.
    clean_df = dp.prepare_phase1_data(raw_df)
    assert len(clean_df) == CLEANED_ROWS
    assert int(clean_df.isna().sum().sum()) == 0
    assert {"HOUR", "TIME_BIN"}.issubset(clean_df.columns)
    assert UNKNOWN_COLLISION_TYPE not in set(clean_df["COLLISION_TYPE"])
    assert set(clean_df["SEVERITY"]).issubset(set(VALID_SEVERITIES))
    assert clean_df["HOUR"].between(0, 23).all()
    assert clean_df["X"].between(X_MIN, X_MAX).all()
    assert clean_df["Y"].between(Y_MIN, Y_MAX).all()

    # Stage 3 — the EDA summaries describe the whole cleaned dataset.
    frequencies = collision_frequency_table(clean_df)
    assert frequencies["COUNT"].sum() == CLEANED_ROWS
    assert frequencies["PERCENT"].sum() == pytest.approx(100.0)
    assert frequencies.index[0] == "Side Swipe"

    severities = severity_distribution(clean_df)
    for collision_type in severities.index:
        assert severities.loc[collision_type].sum() == pytest.approx(100.0)

    # Stage 4 — every cleaned record survives into the modelling matrix.
    features, transformers = prepare_modelling_features(clean_df)
    assert len(features) == CLEANED_ROWS
    assert int(features.isna().sum().sum()) == 0
    assert list(features.index) == list(clean_df.index)
    assert transformers["coordinate_scaler"] is not None

    # Stage 5 — k=2 really does separate better than k=3, which is why the
    # report selects it over the elbow's suggestion.
    selection = evaluate_candidate_clusters(features, k_values=range(1, 4))
    assert list(selection["K"]) == [1, 2, 3]
    assert np.isnan(selection.loc[0, "SILHOUETTE"])
    assert selection.loc[1, "SILHOUETTE"] > selection.loc[2, "SILHOUETTE"]
    assert selection.loc[0, "WCSS"] > selection.loc[1, "WCSS"] > selection.loc[2, "WCSS"]

    # Stage 6 — labelling loses no records and reproduces the reported split.
    clustered_df, model = fit_final_clusters(clean_df, features, n_clusters=2)
    assert len(clustered_df) == CLEANED_ROWS
    assert set(clustered_df["CLUSTER"]) == {1, 2}
    sizes = clustered_df["CLUSTER"].value_counts().sort_index()
    assert sizes.sum() == CLEANED_ROWS
    assert sizes.tolist() == [11551, 3198]

    # Stage 7 — characterization covers every cluster and every category.
    summary = summarize_clusters(clustered_df)
    assert summary["ACCIDENTS"].sum() == CLEANED_ROWS
    assert summary["PERCENT"].sum() == pytest.approx(100.0)
    assert summary["CIRCULAR_MEAN_HOUR"].between(0, 24).all()

    collision_pct = cluster_composition(clustered_df, "COLLISION_TYPE", COLLISION_TYPES)
    for cluster in collision_pct.index:
        assert collision_pct.loc[cluster].sum() == pytest.approx(100.0)
    assert cluster_centroids(model, features.columns).shape == (2, features.shape[1])

    # Stage 8 — the inference stage consumes the cluster labels and its
    # assumptions hold on the real contingency table.
    result = cluster_collision_chi_square(clustered_df)
    assert result["contingency"].values.sum() == CLEANED_ROWS
    assert (result["expected"] >= 5).all().all()
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["degrees_of_freedom"] == 6
    assert result["chi2"] == pytest.approx(138.5411, abs=1e-3)
    assert result["p_value"] < 0.05


# EDSA-ST002
def test_cbadvai_pipeline_end_to_end(raw_df):
    """Real CSV -> cleaning -> encoding -> split -> scaling -> training -> metrics."""
    np.random.seed(42)
    dp = DataPipeline()

    # Stage 1 & 2 — cleaning produces the cyclic hour features this pipeline needs.
    assert raw_df.shape == RAW_SHAPE
    processed = dp.process_data(raw_df)
    assert len(processed) == CLEANED_ROWS
    assert {"HOUR_SIN", "HOUR_COS"}.issubset(processed.columns)

    # Stage 3 — one-hot encoding expands both categorical columns.
    prepared = dp.Onehot_Encoding(processed)
    assert len(prepared) == CLEANED_ROWS
    collision_dummies = [c for c in prepared.columns if c.startswith("COLLISION_TYPE_")]
    severity_dummies = [c for c in prepared.columns if c.startswith("SEVERITY_")]
    assert len(collision_dummies) == len(COLLISION_TYPES)
    assert len(severity_dummies) == len(VALID_SEVERITIES)

    # Stage 4 — feature/target selection picks up the encoded columns.
    chosen_features = ["X", "Y", "HOUR_SIN", "HOUR_COS", "SEVERITY"]
    chosen_target = "COLLISION_TYPE"
    x_columns = [
        col for col in prepared.columns
        if any(col == f or col.startswith(f"{f}_") for f in chosen_features)
    ]
    y_columns = [
        col for col in prepared.columns
        if col == chosen_target or col.startswith(f"{chosen_target}_")
    ]
    assert len(x_columns) == 7
    assert len(y_columns) == len(COLLISION_TYPES)

    X, y = prepared[x_columns], prepared[y_columns]
    # Every row carries exactly one collision-type label after encoding.
    np.testing.assert_array_equal(y.sum(axis=1).unique(), [1.0])

    # Stage 5 — the double split accounts for every row and leaks none.
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.2, random_state=42
    )
    assert [len(X_train), len(X_valid), len(X_test)] == [9439, 2360, 2950]
    assert len(X_train) + len(X_valid) + len(X_test) == CLEANED_ROWS
    train_idx, valid_idx, test_idx = set(X_train.index), set(X_valid.index), set(X_test.index)
    assert not train_idx & test_idx
    assert not train_idx & valid_idx
    assert not valid_idx & test_idx

    # Stage 6 — scaling is fitted on training data only and applied to the rest.
    X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()
    X_valid_np, y_valid_np = X_valid.to_numpy(), y_valid.to_numpy()
    X_test_np, y_test_np = X_test.to_numpy(), y_test.to_numpy()

    scaler = StandardScaler()
    cols_to_scale = [x_columns.index(col) for col in ["X", "Y"]]
    X_train_np[:, cols_to_scale] = scaler.fit_transform(X_train_np[:, cols_to_scale])
    X_valid_np[:, cols_to_scale] = scaler.transform(X_valid_np[:, cols_to_scale])
    X_test_np[:, cols_to_scale] = scaler.transform(X_test_np[:, cols_to_scale])
    assert X_train_np[:, cols_to_scale].mean() == pytest.approx(0.0, abs=1e-9)
    assert X_train_np[:, cols_to_scale].std() == pytest.approx(1.0, abs=1e-6)

    # Stage 7 — a short training run stands in for the full hyperparameter
    # search, which is far too slow for CI. The point is that real data flows
    # through training and early stopping, not that accuracy is maximized.
    model = MultinomialLogisticRegression(
        num_features=X_train_np.shape[1], num_classes=y_train_np.shape[1]
    )
    results, acc_history, loss_history = train_and_evaluate_final_model(
        model, X_train_np, y_train_np, X_valid_np, y_valid_np, X_test_np, y_test_np,
        lr=0.1, max_epochs=8, batch_size=256, patience_limit=8,
    )
    assert len(acc_history) == len(loss_history) > 0
    assert loss_history[-1] < loss_history[0], "training did not reduce loss"
    for key in ("validation_accuracy", "testing_accuracy", "testing_loss"):
        assert key in results
    assert 0.0 <= results["testing_accuracy"] <= 1.0

    # Stage 8 — evaluation consumes the trained model's test-set predictions.
    probabilities = model.forward(X_test_np)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = np.argmax(y_test_np, axis=1)
    assert probabilities.shape == (len(X_test), len(COLLISION_TYPES))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert set(np.unique(predictions)).issubset(set(range(len(COLLISION_TYPES))))

    metrics = summarize_metrics(
        true_labels, predictions, probabilities, y_test_np, "System Test MLR"
    )
    for name in (
        "Accuracy", "Balanced Accuracy", "Precision (macro)",
        "Recall (macro)", "F1-score (macro)", "PR-AUC (macro)",
    ):
        assert 0.0 <= metrics[name] <= 1.0, f"{name} outside [0, 1]"
    assert metrics["Accuracy"] == pytest.approx(results["testing_accuracy"], abs=1e-9)
