import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from models.cluster_plots import (
    CLUSTER_COLORS,
    _cluster_color,
    plot_cluster_collision_heatmap,
    plot_cluster_hourly_distribution,
    plot_cluster_selection,
    plot_cluster_severity_composition,
    plot_clusters_in_coordinate_space,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def clustered_df():
    return pd.DataFrame({
        'X': [120.98, 120.99, 121.06, 121.07],
        'Y': [14.52, 14.53, 14.66, 14.67],
        'HOUR': [8, 9, 20, 21],
        'CLUSTER': [1, 1, 2, 2],
    })


@pytest.fixture
def cluster_summary():
    return pd.DataFrame(
        {'X_MEAN': [120.985, 121.065], 'Y_MEAN': [14.525, 14.665]},
        index=pd.Index([1, 2], name='CLUSTER'),
    )


# EDSA-UT077
def test_plot_cluster_selection_marks_selected_k():
    selection = pd.DataFrame({
        'K': [1, 2, 3],
        'WCSS': [100.0, 60.0, 45.0],
        'SILHOUETTE': [np.nan, 0.55, 0.40],
    })
    figure, axes = plot_cluster_selection(selection, selected_k=2)
    assert isinstance(figure, plt.Figure)
    assert len(axes) == 2
    assert axes[0].get_ylabel() == 'WCSS'
    assert axes[1].get_ylabel() == 'Silhouette score'


# EDSA-UT078
def test_plot_clusters_in_coordinate_space_returns_figure(clustered_df, cluster_summary):
    figure, axis = plot_clusters_in_coordinate_space(clustered_df, cluster_summary)
    assert isinstance(figure, plt.Figure)
    assert axis.get_xlabel() == 'Longitude (X)'
    # One scatter per cluster plus one for the centers.
    assert len(axis.collections) == 3


# EDSA-UT079
def test_plot_cluster_hourly_distribution_draws_one_line_per_cluster():
    hour_pct = pd.DataFrame(
        np.full((2, 24), 100 / 24),
        index=pd.Index([1, 2], name='CLUSTER'),
        columns=range(24),
    )
    figure, axis = plot_cluster_hourly_distribution(hour_pct)
    assert isinstance(figure, plt.Figure)
    assert len(axis.lines) == 2


# EDSA-UT080
def test_plot_cluster_severity_composition_returns_three_panels():
    severity_pct = pd.DataFrame(
        {'Property': [90.0, 80.0], 'Injury': [9.0, 19.0], 'Fatal': [1.0, 1.0]},
        index=pd.Index([1, 2], name='CLUSTER'),
    )
    figure, axes = plot_cluster_severity_composition(severity_pct)
    assert isinstance(figure, plt.Figure)
    assert [axis.get_title() for axis in axes] == ['Property', 'Injury', 'Fatal']


# EDSA-UT081
def test_plot_cluster_collision_heatmap_annotates_every_cell():
    collision_pct = pd.DataFrame(
        {'Rear-End': [40.0, 30.0], 'Side Swipe': [60.0, 70.0]},
        index=pd.Index([1, 2], name='CLUSTER'),
    )
    figure, axis = plot_cluster_collision_heatmap(collision_pct)
    assert isinstance(figure, plt.Figure)
    # One text annotation per cell of the 2x2 table.
    assert len(axis.texts) == 4


# EDSA-UT082
def test_cluster_color_wraps_beyond_the_palette():
    assert _cluster_color(1) == CLUSTER_COLORS[0]
    assert _cluster_color(3) == CLUSTER_COLORS[2]
    # A fourth cluster reuses the first color instead of raising.
    assert _cluster_color(4) == CLUSTER_COLORS[0]
