import numpy as np
import pandas as pd
import pytest

from models.inference import cluster_collision_chi_square


def make_clustered(cluster_labels, collision_types):
    return pd.DataFrame({
        'CLUSTER': cluster_labels,
        'COLLISION_TYPE': collision_types,
    })


# EDSA-UT083
def test_cluster_collision_chi_square_raises_on_missing_columns():
    df = pd.DataFrame({'CLUSTER': [1, 2]})
    with pytest.raises(ValueError, match='COLLISION_TYPE'):
        cluster_collision_chi_square(df)


# EDSA-UT084
def test_cluster_collision_chi_square_detects_independence():
    # Both clusters have an identical 10/10 split, so the variables are independent.
    df = make_clustered(
        [1] * 20 + [2] * 20,
        (['Rear-End'] * 10 + ['Side Swipe'] * 10) * 2,
    )
    result = cluster_collision_chi_square(df)

    assert result['chi2'] == pytest.approx(0.0)
    assert result['p_value'] == pytest.approx(1.0)
    assert result['degrees_of_freedom'] == 1
    np.testing.assert_allclose(result['expected'].values, 10.0)
    np.testing.assert_allclose(result['standardized_residuals'].values, 0.0, atol=1e-12)


# EDSA-UT085
def test_cluster_collision_chi_square_detects_perfect_association():
    # Cluster 1 is entirely Rear-End and cluster 2 entirely Side Swipe.
    df = make_clustered(
        [1] * 20 + [2] * 20,
        ['Rear-End'] * 20 + ['Side Swipe'] * 20,
    )
    result = cluster_collision_chi_square(df)

    # Every expected count is 10, and each of the four cells contributes 10 to chi-square.
    np.testing.assert_allclose(result['expected'].values, 10.0)
    assert result['chi2'] == pytest.approx(40.0)
    assert result['degrees_of_freedom'] == 1
    assert result['p_value'] < 0.05


# EDSA-UT086
def test_cluster_collision_chi_square_returns_aligned_diagnostics():
    df = make_clustered(
        [1] * 9 + [2] * 9,
        (['Rear-End'] * 4 + ['Side Swipe'] * 3 + ['Angle Impact'] * 2)
        + (['Rear-End'] * 2 + ['Side Swipe'] * 3 + ['Angle Impact'] * 4),
    )
    result = cluster_collision_chi_square(df)

    contingency = result['contingency']
    assert contingency.shape == (2, 3)
    assert contingency.values.sum() == 18
    assert result['degrees_of_freedom'] == 2

    # Expected counts and residuals share the contingency table's labels.
    assert list(result['expected'].index) == list(contingency.index)
    assert list(result['expected'].columns) == list(contingency.columns)
    assert result['standardized_residuals'].shape == contingency.shape
    # Row and column totals are preserved by the expected-count calculation.
    np.testing.assert_allclose(
        result['expected'].sum(axis=1).values, contingency.sum(axis=1).values
    )
