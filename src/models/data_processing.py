from pathlib import Path

import numpy as np
import pandas as pd

STUDY_VARIABLES = ['DATETIME_PST', 'SEVERITY', 'X', 'Y', 'COLLISION_TYPE']

VALID_SEVERITIES = ('Property', 'Injury', 'Fatal')

UNKNOWN_COLLISION_TYPE = 'No Collision Stated'
COLLISION_TYPES = ('Angle Impact', 'Head-On', 'Hit Object', 'Multiple', 'Rear-End', 'Self-Accident', 'Side Swipe')

X_MIN, X_MAX = 120.96, 121.08
Y_MIN, Y_MAX = 14.51, 14.68

TIME_BIN_EDGES = [0, 6, 12, 18, 24]
TIME_BIN_LABELS = ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']


class DataPipeline:
    def __init__(self):
        # Stores the final processed dataframe
        self.dataset = None

    # Dataset Pathing
    def find_dataset_path(self, start: Path | None = None) -> Path:
        """Find the accident CSV near this file or in a nearby dataset folder."""
        start = Path.cwd() if start is None else Path(start)

        filename = "RTA_EDSA_2007-2016.csv"
        locations = [start, *start.parents]
        candidates = [location / filename for location in locations]
        candidates += [location / "dataset" / filename for location in locations]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Place {filename} beside this notebook or in a nearby dataset folder."
        )

    def load_dataset(self, csv_path: str | Path) -> pd.DataFrame:
        """Load the raw accident CSV."""
        return pd.read_csv(csv_path)


    # Variable Handling
    def select_study_variables(
        self,
        df: pd.DataFrame,
        variables: list[str] | tuple[str, ...] = STUDY_VARIABLES,
    ) -> pd.DataFrame:
        """Return an independent frame containing only the variables used in the study."""
        missing_columns = [column for column in variables if column not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        return df.loc[:, list(variables)].copy()

    def audit_study_data(self, df: pd.DataFrame) -> dict[str, object]:
        """Summarize missing values, exact duplicates, types, and category values."""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing": df.isna().sum(),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str),
            "severity_values": sorted(df["SEVERITY"].dropna().unique().tolist()),
            "collision_type_values": sorted(
                df["COLLISION_TYPE"].dropna().unique().tolist()
            ),
        }

    def find_apparent_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return all records duplicated across the selected modelling variables."""
        return df[df.duplicated(subset=STUDY_VARIABLES, keep=False)].copy()

    def parse_datetime_features(
        self, df: pd.DataFrame, include_time_bin: bool = False
    ) -> pd.DataFrame:
        """Parse Philippine timestamps and derive HOUR, optionally with four time-of-day bins."""
        result = df.copy()
        result["DATETIME_PST"] = pd.to_datetime(
            result["DATETIME_PST"], format="mixed", errors="raise"
        )
        result["HOUR"] = result["DATETIME_PST"].dt.hour
        if include_time_bin:
            result["TIME_BIN"] = pd.cut(
                result["HOUR"],
                bins=TIME_BIN_EDGES,
                labels=TIME_BIN_LABELS,
                right=False,
                include_lowest=True,
            )
        return result

    def find_invalid_coordinate_rows(self,
        df: pd.DataFrame,
        x_min: float = X_MIN,
        x_max: float = X_MAX,
        y_min: float = Y_MIN,
        y_max: float = Y_MAX,
    ) -> pd.DataFrame:
        """Return records outside the declared geographic bounds or with null coordinates."""
        invalid = (
            df["X"].isna()
            | df["Y"].isna()
            | ~df["X"].between(x_min, x_max, inclusive="both")
            | ~df["Y"].between(y_min, y_max, inclusive="both")
        )
        return df.loc[invalid].copy()

    def filter_known_collision_types(
        self,
        df: pd.DataFrame,
        unknown_label: str = UNKNOWN_COLLISION_TYPE,
    ) -> pd.DataFrame:
        """Remove records without a stated collision type."""
        return df.loc[df["COLLISION_TYPE"].ne(unknown_label)].copy()

    def validate_categories(self, df: pd.DataFrame) -> None:
        """Raise an error when severity or known collision labels are unexpected."""
        invalid_severity = set(df["SEVERITY"].dropna()) - set(VALID_SEVERITIES)
        allowed_collisions = set(COLLISION_TYPES) | {UNKNOWN_COLLISION_TYPE}
        invalid_collision = set(df["COLLISION_TYPE"].dropna()) - allowed_collisions

        if invalid_severity:
            raise ValueError(f"Unexpected severity values: {sorted(invalid_severity)}")
        if invalid_collision:
            raise ValueError(
                f"Unexpected collision-type values: {sorted(invalid_collision)}"
            )

    def hour_converstion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert hour to sine and cosine features"""

        hour_angle = 2 * np.pi * df["HOUR"] / 24
        df["HOUR_SIN"] = np.sin(hour_angle)
        df["HOUR_COS"] = np.cos(hour_angle)

        return df

    def _select_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared first stage of both pipelines: select the study variables and validate them."""
        result = self.select_study_variables(df)
        self.validate_categories(result)

        if result[STUDY_VARIABLES].isna().any().any():
            raise ValueError("Study variables contain missing values.")
        if not self.find_invalid_coordinate_rows(result).empty:
            raise ValueError("Study variables contain coordinates outside EDSA bounds.")
        return result

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """CBADVAI pipeline: validate, derive HOUR and cyclic hour features, drop unknown collisions."""
        result = self._select_and_validate(df)
        result = self.parse_datetime_features(result)
        result = self.filter_known_collision_types(result)
        result = self.hour_converstion(result)
        return result.reset_index(drop=True)

    def prepare_phase1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """CBDATSI Phase 1 pipeline: validate, derive HOUR and TIME_BIN, drop unknown collisions."""
        result = self._select_and_validate(df)
        result = self.parse_datetime_features(result, include_time_bin=True)
        result = self.filter_known_collision_types(result)
        return result.reset_index(drop=True)

    def Onehot_Encoding(self, df: pd.DataFrame):
        """ One-Hot Encoding """
        df = pd.get_dummies(df, columns=["COLLISION_TYPE"], dtype=float)
        df = pd.get_dummies(df, columns=["SEVERITY"], dtype=float)
        return df
