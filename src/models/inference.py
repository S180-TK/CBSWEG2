"""Statistical inference for the CBDATSI Phase 2 cluster analysis."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cluster_collision_chi_square(data):
    """Test cluster and collision-type association and return diagnostics."""
    required = {"CLUSTER", "COLLISION_TYPE"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    contingency = pd.crosstab(data["CLUSTER"], data["COLLISION_TYPE"])
    chi2, p_value, degrees_of_freedom, expected = chi2_contingency(
        contingency,
        correction=False,
    )
    expected = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns,
    )
    standardized_residuals = (contingency - expected) / np.sqrt(expected)

    return {
        "contingency": contingency,
        "expected": expected,
        "standardized_residuals": standardized_residuals,
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
    }
