"""Visualizations for the CBDATSI Phase 1 exploratory data analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.eda import hourly_collision_counts


def plot_collision_frequency(
    frequency_table: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot collision-type counts."""
    ordered = frequency_table.sort_values("COUNT")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(ordered.index, ordered["COUNT"], color="#3B82A0")
    axis.set_xlabel("Recorded accidents")
    axis.set_ylabel("Collision type")
    axis.set_title("Recorded EDSA Accidents by Collision Type (2007–2016)")
    figure.tight_layout()
    return figure, axis


def plot_coordinate_density(
    df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot accident density over the raw longitude and latitude coordinates."""
    figure, axis = plt.subplots(figsize=(8, 7))
    density = axis.hexbin(
        df["X"],
        df["Y"],
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axis.set_xlabel("Longitude (X)")
    axis.set_ylabel("Latitude (Y)")
    axis.set_title("Density of Recorded Accidents Along EDSA")
    axis.set_aspect(1 / np.cos(np.deg2rad(df["Y"].mean())))
    figure.colorbar(
        density,
        ax=axis,
        label="Recorded accident count (logarithmic color scale)",
    )
    figure.tight_layout()
    return figure, axis


def plot_hourly_collision_heatmap(
    df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the hourly percentage distribution within each collision type."""
    counts = hourly_collision_counts(df).T
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    figure, axis = plt.subplots(figsize=(13, 5))
    image = axis.imshow(percentages.values, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(24))
    axis.set_xticklabels(range(24), fontsize=8)
    axis.set_yticks(range(len(percentages.index)))
    axis.set_yticklabels(percentages.index)
    axis.set_xlabel("Hour of day")
    axis.set_title("Hourly Distribution Within Each Collision Type")
    figure.colorbar(image, ax=axis, label="% within collision type")
    figure.tight_layout()
    return figure, axis


def plot_severity_distribution(
    severity_percentages: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot each severity percentage in a separate panel with its own scale."""
    colors = ["#4C78A8", "#F2A541", "#C83349"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))

    for axis, severity, color in zip(
        axes, severity_percentages.columns, colors
    ):
        ordered = severity_percentages[severity].sort_values()
        axis.barh(ordered.index, ordered.values, color=color)
        axis.set_xlabel("Percentage within collision type")
        axis.set_title(severity)

    axes[0].set_ylabel("Collision type")
    figure.suptitle(
        "Severity Composition by Collision Type — Independent Scales",
        fontsize=14,
    )
    figure.tight_layout()
    return figure, axes
