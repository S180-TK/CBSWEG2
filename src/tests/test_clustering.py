import numpy as np
import pandas as pd
import pytest

from models.clustering import (
    circular_mean_hour,
    cluster_centroids,
    cluster_composition,
    evaluate_candidate_clusters,
    fit_final_clusters,
    prepare_modelling_features,
    summarize_clusters,
)
from models.data_processing import COLLISION_TYPES, VALID_SEVERITIES

EXPECTED_BASE_FEATURES = [
    'X_SCALED', 'Y_SCALED', 'HOUR_SIN', 'HOUR_COS',
    'SEVERITY_PROPERTY', 'SEVERITY_INJURY', 'SEVERITY_FATAL',
]


@pytest.fixture
def cluster_df():
    """Twenty accidents in two clearly separated spatial groups."""
    size = 10
    south = pd.DataFrame({
        'X': np.linspace(120.98, 120.99, size),
        'Y': np.linspace(14.52, 14.53, size),
        'HOUR': [8] * size,
        'SEVERITY': ['Property'] * size,
        'COLLISION_TYPE': ['Rear-End'] * size,
        'DATETIME_PST': pd.to_datetime(['2015-06-01 08:00:00'] * size),
    })
    north = pd.DataFrame({
        'X': np.linspace(121.06, 121.07, size),
        'Y': np.linspace(14.66, 14.67, size),
        'HOUR': [20] * size,
        'SEVERITY': ['Injury'] * size,
        'COLLISION_TYPE': ['Side Swipe'] * size,
        'DATETIME_PST': pd.to_datetime(['2016-06-01 20:00:00'] * size),
    })
    return pd.concat([south, north], ignore_index=True)


# EDSA-UT065
def test_prepare_modelling_features_builds_expected_columns(cluster_df):
    features, transformers = prepare_modelling_features(cluster_df)
    assert list(features.columns) == EXPECTED_BASE_FEATURES
    assert len(features) == len(cluster_df)
    assert int(features.isna().sum().sum()) == 0
    # Fatal never occurs in the fixture, so the reindex must fill it with zeros.
    assert features['SEVERITY_FATAL'].eq(0.0).all()
    assert transformers['year_scaler'] is None


# EDSA-UT066
def test_prepare_modelling_features_standardizes_coordinates(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    assert features['X_SCALED'].mean() == pytest.approx(0.0, abs=1e-9)
    assert features['X_SCALED'].std(ddof=0) == pytest.approx(1.0)
    assert features['Y_SCALED'].mean() == pytest.approx(0.0, abs=1e-9)
    # Cyclic hour features stay on the unit circle rather than being standardized.
    circle = features['HOUR_SIN'] ** 2 + features['HOUR_COS'] ** 2
    np.testing.assert_allclose(circle, 1.0, atol=1e-9)


# EDSA-UT067
def test_prepare_modelling_features_adds_scaled_year_when_requested(cluster_df):
    features, transformers = prepare_modelling_features(cluster_df, include_year=True)
    assert list(features.columns) == EXPECTED_BASE_FEATURES + ['YEAR_SCALED']
    assert transformers['year_scaler'] is not None
    assert features['YEAR_SCALED'].mean() == pytest.approx(0.0, abs=1e-9)


# EDSA-UT068
def test_evaluate_candidate_clusters_reports_metrics_per_k(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    results = evaluate_candidate_clusters(features, k_values=range(1, 4))

    assert list(results['K']) == [1, 2, 3]
    assert list(results.columns) == [
        'K', 'WCSS', 'WCSS_REDUCTION_PERCENT', 'SILHOUETTE',
        'MIN_CLUSTER_SIZE', 'MAX_CLUSTER_SIZE',
    ]
    # k=1 has no between-cluster separation, so silhouette is undefined.
    assert np.isnan(results.loc[0, 'SILHOUETTE'])
    assert np.isnan(results.loc[0, 'WCSS_REDUCTION_PERCENT'])
    assert not np.isnan(results.loc[1, 'SILHOUETTE'])
    assert results['MIN_CLUSTER_SIZE'].iloc[1] + results['MAX_CLUSTER_SIZE'].iloc[1] == 20


# EDSA-UT069
def test_evaluate_candidate_clusters_wcss_decreases_with_more_clusters(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    results = evaluate_candidate_clusters(features, k_values=range(1, 4))
    wcss = results['WCSS'].tolist()
    assert wcss[0] > wcss[1] > wcss[2]


# EDSA-UT070
def test_circular_mean_hour_wraps_across_midnight():
    # A naive arithmetic mean of 23 and 1 would give 12; the circular mean gives 0.
    assert circular_mean_hour(pd.Series([23, 1])) == pytest.approx(0.0, abs=1e-9)


# EDSA-UT071
def test_circular_mean_hour_matches_plain_mean_away_from_midnight():
    assert circular_mean_hour(pd.Series([11, 13])) == pytest.approx(12.0)
    assert circular_mean_hour(pd.Series([8, 8, 8])) == pytest.approx(8.0)


# EDSA-UT072
def test_fit_final_clusters_attaches_one_based_labels(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    clustered, model = fit_final_clusters(cluster_df, features, n_clusters=2)

    assert set(clustered['CLUSTER'].unique()) == {1, 2}
    assert model.n_clusters == 2
    assert len(clustered) == len(cluster_df)
    # The original records are preserved alongside the new label.
    assert {'X', 'Y', 'HOUR', 'SEVERITY', 'COLLISION_TYPE'}.issubset(clustered.columns)


# EDSA-UT073
def test_fit_final_clusters_separates_well_separated_groups(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    clustered, _ = fit_final_clusters(cluster_df, features, n_clusters=2)

    south_labels = set(clustered.loc[:9, 'CLUSTER'])
    north_labels = set(clustered.loc[10:, 'CLUSTER'])
    assert len(south_labels) == 1
    assert len(north_labels) == 1
    assert south_labels != north_labels


# EDSA-UT074
def test_summarize_clusters_reports_size_and_centers(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    clustered, _ = fit_final_clusters(cluster_df, features, n_clusters=2)
    summary = summarize_clusters(clustered)

    assert summary['ACCIDENTS'].sum() == 20
    assert summary['PERCENT'].sum() == pytest.approx(100.0)
    assert {'X_MEAN', 'Y_MEAN', 'X_STD', 'Y_STD', 'MODAL_HOUR',
            'CIRCULAR_MEAN_HOUR'}.issubset(summary.columns)
    # Each group has a single constant hour, so both summaries agree on it.
    for cluster in summary.index:
        assert summary.loc[cluster, 'MODAL_HOUR'] == pytest.approx(
            summary.loc[cluster, 'CIRCULAR_MEAN_HOUR'], abs=1e-6
        )


# EDSA-UT075
def test_cluster_composition_returns_within_cluster_percentages(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    clustered, _ = fit_final_clusters(cluster_df, features, n_clusters=2)

    severity = cluster_composition(clustered, 'SEVERITY', VALID_SEVERITIES)
    assert list(severity.columns) == list(VALID_SEVERITIES)
    for cluster in severity.index:
        assert severity.loc[cluster].sum() == pytest.approx(100.0)

    collisions = cluster_composition(clustered, 'COLLISION_TYPE', COLLISION_TYPES)
    assert list(collisions.columns) == list(COLLISION_TYPES)
    # Collision types absent from the fixture are filled with zero rather than dropped.
    assert collisions['Head-On'].eq(0).all()


# EDSA-UT076
def test_cluster_centroids_are_indexed_from_one(cluster_df):
    features, _ = prepare_modelling_features(cluster_df)
    _, model = fit_final_clusters(cluster_df, features, n_clusters=2)
    centroids = cluster_centroids(model, features.columns)

    assert centroids.shape == (2, len(EXPECTED_BASE_FEATURES))
    assert list(centroids.index) == [1, 2]
    assert centroids.index.name == 'CLUSTER'
    assert list(centroids.columns) == EXPECTED_BASE_FEATURES
