import pandas as pd
import pytest

from models.data_processing import COLLISION_TYPES, TIME_BIN_LABELS, VALID_SEVERITIES
from models.eda import (
    collision_count_statistics,
    collision_frequency_table,
    densest_coordinate_cell,
    hour_statistics,
    hourly_collision_counts,
    hourly_pattern_correlations,
    modal_hour_by_collision,
    overall_coordinate_correlation,
    overall_coordinate_statistics,
    severity_count_correlations,
    severity_distribution,
    time_distribution,
)


@pytest.fixture
def eda_df():
    """Six accidents: 3 Rear-End, 2 Side Swipe, 1 Head-On, with hand-checkable values."""
    return pd.DataFrame({
        'COLLISION_TYPE': [
            'Rear-End', 'Rear-End', 'Rear-End',
            'Side Swipe', 'Side Swipe', 'Head-On',
        ],
        'SEVERITY': [
            'Property', 'Property', 'Injury',
            'Property', 'Injury', 'Fatal',
        ],
        'X': [121.00, 121.01, 121.02, 121.03, 121.04, 121.05],
        'Y': [14.55, 14.56, 14.57, 14.58, 14.59, 14.60],
        'HOUR': [3, 9, 9, 15, 21, 3],
        'TIME_BIN': [
            'Night (0-6)', 'Morning (6-12)', 'Morning (6-12)',
            'Afternoon (12-18)', 'Evening (18-24)', 'Night (0-6)',
        ],
    })


# EDSA-UT049
def test_collision_frequency_table_counts_and_percentages(eda_df):
    result = collision_frequency_table(eda_df)
    assert list(result.index) == ['Rear-End', 'Side Swipe', 'Head-On']
    assert list(result['COUNT']) == [3, 2, 1]
    assert result['PERCENT'].iloc[0] == pytest.approx(50.0)
    assert result['PERCENT'].sum() == pytest.approx(100.0)


# EDSA-UT050
def test_collision_count_statistics_matches_hand_computed_values(eda_df):
    result = collision_count_statistics(collision_frequency_table(eda_df))
    assert result['mean'] == pytest.approx(2.0)
    assert result['median'] == pytest.approx(2.0)
    assert result['standard_deviation'] == pytest.approx(1.0)
    assert result['minimum'] == 1
    assert result['maximum'] == 3
    assert result['range'] == 2
    assert result['iqr'] == pytest.approx(1.0)


# EDSA-UT051
def test_overall_coordinate_statistics_summarizes_both_axes(eda_df):
    result = overall_coordinate_statistics(eda_df)
    assert list(result.index) == ['X', 'Y']
    assert result.loc['X', 'count'] == 6
    assert result.loc['X', 'mean'] == pytest.approx(121.025)
    assert result.loc['Y', 'mean'] == pytest.approx(14.575)
    assert result.loc['X', 'range'] == pytest.approx(0.05)
    assert result.loc['Y', 'range'] == pytest.approx(0.05)


# EDSA-UT052
def test_overall_coordinate_correlation_detects_perfect_relationship(eda_df):
    # X and Y both increase by a constant step, so they are perfectly correlated.
    assert overall_coordinate_correlation(eda_df) == pytest.approx(1.0)


# EDSA-UT053
def test_densest_coordinate_cell_locates_most_populated_grid_cell(eda_df):
    result = densest_coordinate_cell(eda_df, bins=2)
    # The lower-left 2x2 cell holds the first three accidents.
    assert result['count'] == 3
    assert result['x_min'] <= result['x_center'] <= result['x_max']
    assert result['y_min'] <= result['y_center'] <= result['y_max']


# EDSA-UT054
def test_hour_statistics_describes_hour_per_collision_type(eda_df):
    result = hour_statistics(eda_df)
    assert result.loc['Rear-End', 'count'] == 3
    assert result.loc['Rear-End', 'mean'] == pytest.approx(7.0)
    assert result.loc['Side Swipe', 'mean'] == pytest.approx(18.0)
    assert result.loc['Head-On', 'count'] == 1


# EDSA-UT055
def test_modal_hour_by_collision_returns_earliest_mode(eda_df):
    result = modal_hour_by_collision(eda_df)
    assert result['Rear-End'] == 9
    assert result['Head-On'] == 3
    # Side Swipe hours 15 and 21 tie, so the earliest is returned.
    assert result['Side Swipe'] == 15


# EDSA-UT056
def test_time_distribution_returns_within_type_percentages(eda_df):
    result = time_distribution(eda_df)
    assert list(result.columns) == list(TIME_BIN_LABELS)
    assert result.loc['Head-On', 'Night (0-6)'] == pytest.approx(100.0)
    assert result.loc['Rear-End', 'Morning (6-12)'] == pytest.approx(200 / 3)
    assert result.loc['Side Swipe', 'Afternoon (12-18)'] == pytest.approx(50.0)
    for collision_type in result.index:
        assert result.loc[collision_type].sum() == pytest.approx(100.0)


# EDSA-UT057
def test_hourly_collision_counts_covers_all_hours_and_types(eda_df):
    result = hourly_collision_counts(eda_df)
    assert result.shape == (24, len(COLLISION_TYPES))
    assert list(result.index) == list(range(24))
    assert result.loc[9, 'Rear-End'] == 2
    assert result.loc[3, 'Head-On'] == 1
    assert result.values.sum() == 6


# EDSA-UT058
def test_hourly_pattern_correlations_returns_square_matrix(eda_df):
    result = hourly_pattern_correlations(eda_df)
    assert result.shape == (len(COLLISION_TYPES), len(COLLISION_TYPES))
    # Collision types that actually occur correlate perfectly with themselves.
    for collision_type in ['Rear-End', 'Side Swipe', 'Head-On']:
        assert result.loc[collision_type, collision_type] == pytest.approx(1.0)


# EDSA-UT059
def test_severity_distribution_returns_within_type_percentages(eda_df):
    result = severity_distribution(eda_df)
    assert list(result.columns) == list(VALID_SEVERITIES)
    assert result.loc['Head-On', 'Fatal'] == pytest.approx(100.0)
    assert result.loc['Rear-End', 'Property'] == pytest.approx(200 / 3)
    assert result.loc['Side Swipe', 'Injury'] == pytest.approx(50.0)
    for collision_type in result.index:
        assert result.loc[collision_type].sum() == pytest.approx(100.0)


# EDSA-UT060
def test_severity_count_correlations_returns_square_matrix(eda_df):
    result = severity_count_correlations(eda_df)
    assert result.shape == (len(VALID_SEVERITIES), len(VALID_SEVERITIES))
    assert list(result.columns) == list(VALID_SEVERITIES)
