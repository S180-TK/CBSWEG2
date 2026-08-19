import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from models.eda import collision_frequency_table, severity_distribution
from models.eda_plots import (
    plot_collision_frequency,
    plot_coordinate_density,
    plot_hourly_collision_heatmap,
    plot_severity_distribution,
)


@pytest.fixture
def eda_df():
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
    })


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


# EDSA-UT061
def test_plot_collision_frequency_returns_figure_and_axis(eda_df):
    figure, axis = plot_collision_frequency(collision_frequency_table(eda_df))
    assert isinstance(figure, plt.Figure)
    assert axis.get_xlabel() == 'Recorded accidents'


# EDSA-UT062
def test_plot_coordinate_density_returns_figure_and_axis(eda_df):
    figure, axis = plot_coordinate_density(eda_df)
    assert isinstance(figure, plt.Figure)
    assert axis.get_xlabel() == 'Longitude (X)'


# EDSA-UT063
def test_plot_hourly_collision_heatmap_returns_figure_and_axis(eda_df):
    figure, axis = plot_hourly_collision_heatmap(eda_df)
    assert isinstance(figure, plt.Figure)
    assert len(axis.get_xticks()) == 24


# EDSA-UT064
def test_plot_severity_distribution_returns_three_panels(eda_df):
    figure, axes = plot_severity_distribution(severity_distribution(eda_df))
    assert isinstance(figure, plt.Figure)
    assert len(axes) == 3
    assert [axis.get_title() for axis in axes] == ['Property', 'Injury', 'Fatal']
