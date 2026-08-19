"""Descriptive statistics for the CBDATSI Phase 1 exploratory data analysis."""

import numpy as np
import pandas as pd

from models.data_processing import COLLISION_TYPES, TIME_BIN_LABELS, VALID_SEVERITIES


def collision_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return collision counts and percentages in descending frequency order."""
    counts = df["COLLISION_TYPE"].value_counts()
    result = counts.rename("COUNT").to_frame()
    result["PERCENT"] = result["COUNT"] / result["COUNT"].sum() * 100
    return result


def collision_count_statistics(frequency_table: pd.DataFrame) -> pd.Series:
    """Return central-tendency and dispersion statistics across class counts."""
    counts = frequency_table["COUNT"]
    return pd.Series(
        {
            "mean": counts.mean(),
            "median": counts.median(),
            "mode": counts.mode().iloc[0],
            "standard_deviation": counts.std(),
            "minimum": counts.min(),
            "maximum": counts.max(),
            "range": counts.max() - counts.min(),
            "iqr": counts.quantile(0.75) - counts.quantile(0.25),
        }
    )


def overall_coordinate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the center and spread of the raw coordinates."""
    summary = df[["X", "Y"]].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).T
    summary["range"] = summary["max"] - summary["min"]
    summary["iqr"] = (
        df[["X", "Y"]].quantile(0.75) - df[["X", "Y"]].quantile(0.25)
    )
    return summary


def overall_coordinate_correlation(df: pd.DataFrame) -> float:
    """Return the Pearson correlation between raw longitude and latitude."""
    return float(df["X"].corr(df["Y"]))


def densest_coordinate_cell(df: pd.DataFrame, bins: int = 30) -> pd.Series:
    """Locate the most populated cell in an equal-width X-Y grid."""
    counts, x_edges, y_edges = np.histogram2d(df["X"], df["Y"], bins=bins)
    x_index, y_index = np.unravel_index(np.argmax(counts), counts.shape)
    return pd.Series(
        {
            "count": int(counts[x_index, y_index]),
            "x_center": (x_edges[x_index] + x_edges[x_index + 1]) / 2,
            "y_center": (y_edges[y_index] + y_edges[y_index + 1]) / 2,
            "x_min": x_edges[x_index],
            "x_max": x_edges[x_index + 1],
            "y_min": y_edges[y_index],
            "y_max": y_edges[y_index + 1],
        }
    )


def hour_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize accident hour by collision type."""
    return df.groupby("COLLISION_TYPE")["HOUR"].describe()


def modal_hour_by_collision(df: pd.DataFrame) -> pd.Series:
    """Return the earliest modal accident hour for every collision type."""
    return df.groupby("COLLISION_TYPE")["HOUR"].agg(
        lambda values: values.mode().iloc[0]
    )


def time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return within-collision percentages across the four time bins."""
    table = pd.crosstab(
        df["COLLISION_TYPE"], df["TIME_BIN"], normalize="index"
    ) * 100
    return table.reindex(columns=TIME_BIN_LABELS, fill_value=0)


def hourly_collision_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return hourly counts for every collision type."""
    table = pd.crosstab(df["HOUR"], df["COLLISION_TYPE"])
    return table.reindex(index=range(24), columns=COLLISION_TYPES, fill_value=0)


def hourly_pattern_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlations between collision types' hourly counts."""
    return hourly_collision_counts(df).corr()


def severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return within-collision severity percentages."""
    table = pd.crosstab(
        df["COLLISION_TYPE"], df["SEVERITY"], normalize="index"
    ) * 100
    return table.reindex(columns=VALID_SEVERITIES, fill_value=0)


def severity_count_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlations between severity count patterns across collision types."""
    table = pd.crosstab(df["COLLISION_TYPE"], df["SEVERITY"])
    table = table.reindex(columns=VALID_SEVERITIES, fill_value=0)
    return table.corr()
