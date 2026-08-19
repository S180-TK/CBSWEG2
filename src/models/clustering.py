"""K-means clustering pipeline for the CBDATSI Phase 2 accident-profile analysis."""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

SEVERITY_FEATURE_ORDER = ["Property", "Injury", "Fatal"]


def prepare_modelling_features(df, include_year=False):
    """Create K-means features while preserving the original accident records."""
    features = pd.DataFrame(index=df.index)

    coordinate_scaler = StandardScaler()
    features[["X_SCALED", "Y_SCALED"]] = coordinate_scaler.fit_transform(
        df[["X", "Y"]]
    )

    hour_angle = 2 * np.pi * df["HOUR"] / 24
    features["HOUR_SIN"] = np.sin(hour_angle)
    features["HOUR_COS"] = np.cos(hour_angle)

    severity = pd.get_dummies(df["SEVERITY"], dtype=float).reindex(
        columns=SEVERITY_FEATURE_ORDER, fill_value=0.0
    )
    severity.columns = [f"SEVERITY_{name.upper()}" for name in severity.columns]
    features = pd.concat([features, severity], axis=1)

    year_scaler = None
    if include_year:
        year_scaler = StandardScaler()
        years = df["DATETIME_PST"].dt.year.to_frame(name="YEAR")
        features["YEAR_SCALED"] = year_scaler.fit_transform(years).ravel()

    transformers = {
        "coordinate_scaler": coordinate_scaler,
        "year_scaler": year_scaler,
    }
    return features, transformers


def evaluate_candidate_clusters(
    features, k_values=range(1, 11), silhouette_sample=3000, random_state=42
):
    """Fit candidate K-means models and return reproducible selection metrics."""
    records = []
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            tol=1e-4,
            random_state=random_state,
        )
        labels = model.fit_predict(features)
        cluster_sizes = pd.Series(labels).value_counts()
        score = np.nan
        if k >= 2:
            score = silhouette_score(
                features,
                labels,
                sample_size=min(silhouette_sample, len(features)),
                random_state=random_state,
            )
        records.append(
            {
                "K": k,
                "WCSS": model.inertia_,
                "SILHOUETTE": score,
                "MIN_CLUSTER_SIZE": cluster_sizes.min(),
                "MAX_CLUSTER_SIZE": cluster_sizes.max(),
            }
        )
    results = pd.DataFrame(records)
    results.insert(
        2,
        "WCSS_REDUCTION_PERCENT",
        (results["WCSS"].shift(1) - results["WCSS"])
        / results["WCSS"].shift(1)
        * 100,
    )
    return results


def circular_mean_hour(hours):
    """Return the circular mean of clock hours on a 24-hour cycle."""
    angles = 2 * np.pi * hours / 24
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    hour = (mean_angle % (2 * np.pi)) * 24 / (2 * np.pi)
    # A mean angle just below 2*pi (floating-point noise around midnight) would
    # otherwise report hour 24, which is not a valid clock hour.
    return float(hour % 24)


def fit_final_clusters(df, features, n_clusters, random_state=42):
    """Fit the final K-means model and attach 1-based CLUSTER labels to the records."""
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        tol=1e-4,
        random_state=random_state,
    )
    clustered = df.copy()
    clustered["CLUSTER"] = model.fit_predict(features) + 1
    return clustered, model


def summarize_clusters(clustered_df):
    """Report size, coordinate center and spread, and hourly center for every cluster."""
    summary = clustered_df.groupby("CLUSTER").agg(
        ACCIDENTS=("CLUSTER", "size"),
        X_MEAN=("X", "mean"),
        Y_MEAN=("Y", "mean"),
        X_STD=("X", "std"),
        Y_STD=("Y", "std"),
        MODAL_HOUR=("HOUR", lambda values: values.mode().iloc[0]),
    )
    summary.insert(
        1, "PERCENT", summary["ACCIDENTS"] / len(clustered_df) * 100
    )
    summary["CIRCULAR_MEAN_HOUR"] = clustered_df.groupby("CLUSTER")[
        "HOUR"
    ].apply(circular_mean_hour)
    return summary


def cluster_composition(clustered_df, column, categories):
    """Return within-cluster percentages for a categorical or hour column."""
    table = pd.crosstab(
        clustered_df["CLUSTER"], clustered_df[column], normalize="index"
    ) * 100
    return table.reindex(columns=list(categories), fill_value=0)


def cluster_centroids(model, feature_columns):
    """Return the fitted centroids in the modelling feature space, indexed from 1."""
    centroids = pd.DataFrame(
        model.cluster_centers_,
        columns=list(feature_columns),
        index=range(1, model.n_clusters + 1),
    )
    centroids.index.name = "CLUSTER"
    return centroids
