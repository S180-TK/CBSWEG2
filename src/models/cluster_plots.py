"""Visualizations for the CBDATSI Phase 2 clustering results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLUSTER_COLORS = ["#3B82A0", "#E39C37", "#7A68A6"]
SEVERITY_COLORS = ["#4C78A8", "#F2A541", "#C83349"]


def _cluster_color(cluster: int) -> str:
    """Return a stable color for a 1-based cluster label, wrapping if k exceeds the palette."""
    return CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]


def plot_cluster_selection(
    selection_results: pd.DataFrame, selected_k: int
) -> tuple[plt.Figure, np.ndarray]:
    """Plot WCSS and silhouette scores side by side, marking the selected k."""
    figure, (wcss_axis, silhouette_axis) = plt.subplots(1, 2, figsize=(12, 4.5))

    wcss_axis.plot(selection_results["K"], selection_results["WCSS"], marker="o")
    wcss_axis.axvline(
        selected_k, color="crimson", linestyle="--", label=f"Selected k={selected_k}"
    )
    wcss_axis.set(
        title="Elbow Method — Selected Features",
        xlabel="Number of clusters (k)",
        ylabel="WCSS",
    )
    wcss_axis.legend()

    silhouette_axis.plot(
        selection_results["K"], selection_results["SILHOUETTE"], marker="o"
    )
    silhouette_axis.axvline(
        selected_k, color="crimson", linestyle="--", label=f"Selected k={selected_k}"
    )
    silhouette_axis.set(
        title="Silhouette Method — Selected Features",
        xlabel="Number of clusters (k)",
        ylabel="Silhouette score",
    )
    silhouette_axis.legend()

    figure.suptitle("Selecting the Number of K-means Clusters")
    figure.tight_layout()
    return figure, np.array([wcss_axis, silhouette_axis])


def plot_clusters_in_coordinate_space(
    clustered_df: pd.DataFrame, cluster_summary: pd.DataFrame
) -> tuple[plt.Figure, plt.Axes]:
    """Plot every accident in raw coordinate space, colored by cluster, with centers marked."""
    figure, axis = plt.subplots(figsize=(8, 7))
    for cluster in cluster_summary.index:
        subset = clustered_df[clustered_df["CLUSTER"].eq(cluster)]
        axis.scatter(
            subset["X"], subset["Y"], s=7, alpha=0.25,
            color=_cluster_color(cluster), label=f"Cluster {cluster}"
        )
    axis.scatter(
        cluster_summary["X_MEAN"], cluster_summary["Y_MEAN"],
        marker="X", s=140, color="black", label="Cluster center"
    )
    axis.set(
        title="Final Clusters in Raw Coordinate Space",
        xlabel="Longitude (X)", ylabel="Latitude (Y)"
    )
    axis.set_aspect(1 / np.cos(np.deg2rad(clustered_df["Y"].mean())))
    axis.legend(bbox_to_anchor=(1.4, 0.5), loc="center right")
    figure.tight_layout()
    return figure, axis


def plot_cluster_hourly_distribution(
    cluster_hour_pct: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the hourly percentage profile of every cluster."""
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for cluster in cluster_hour_pct.index:
        axis.plot(
            cluster_hour_pct.columns, cluster_hour_pct.loc[cluster],
            marker="o", linewidth=1.8, color=_cluster_color(cluster),
            label=f"Cluster {cluster}"
        )
    axis.set(
        title="Hourly Distribution Within Each Cluster",
        xlabel="Hour of day", ylabel="Percentage within cluster"
    )
    axis.set_xticks(range(24))
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_cluster_severity_composition(
    cluster_severity_pct: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot each severity percentage per cluster in its own panel with an independent scale."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cluster_names = [f"Cluster {cluster}" for cluster in cluster_severity_pct.index]
    for axis, severity_name, color in zip(
        axes, cluster_severity_pct.columns, SEVERITY_COLORS
    ):
        values = cluster_severity_pct[severity_name]
        axis.barh(cluster_names, values, color=color)
        axis.set(title=severity_name, xlabel="Percentage within cluster")
    figure.suptitle("Severity Composition by Cluster — Independent Scales")
    figure.tight_layout()
    return figure, axes


def plot_cluster_collision_heatmap(
    cluster_collision_pct: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of collision-type percentages within each cluster."""
    figure, axis = plt.subplots(figsize=(12, 4.5))
    image = axis.imshow(cluster_collision_pct.values, aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(cluster_collision_pct.columns)))
    axis.set_xticklabels(cluster_collision_pct.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(cluster_collision_pct.index)))
    axis.set_yticklabels(
        [f"Cluster {cluster}" for cluster in cluster_collision_pct.index]
    )
    axis.set_title("Collision-Type Composition Within Each Cluster (%)")
    for row in range(cluster_collision_pct.shape[0]):
        for column in range(cluster_collision_pct.shape[1]):
            value = cluster_collision_pct.iloc[row, column]
            text_color = (
                "white" if value > cluster_collision_pct.values.max() / 2 else "black"
            )
            axis.text(
                column, row, f"{value:.1f}",
                ha="center", va="center", color=text_color, fontsize=8
            )
    figure.colorbar(image, ax=axis, label="Percentage within cluster")
    figure.tight_layout()
    return figure, axis
