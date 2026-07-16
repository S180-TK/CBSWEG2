import re

import pandas as pd

KEEP_VARIABLES = ["ADDRESS", "DATETIME_PST", "SEVERITY", "X", "Y", "COLLISION_TYPE"]

X_MIN, X_MAX = 120.96, 121.08
Y_MIN, Y_MAX = 14.51, 14.68

HOUR_BIN_EDGES = [0, 6, 12, 18, 24]
HOUR_BIN_LABELS = ["Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"]

SEVERITY_MAP = {"Property": 1, "Injury": 2, "Fatal": 3}

VALID_SEVERITY_VALUES = frozenset(SEVERITY_MAP)
VALID_COLLISION_TYPE_VALUES = frozenset({
    "Angle Impact",
    "Head-On",
    "Hit Object",
    "Multiple",
    "No Collision Stated",
    "Rear-End",
    "Self-Accident",
    "Side Swipe",
})


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def filter_columns(df, keep_variables=KEEP_VARIABLES):
    return df.drop(columns=[col for col in df.columns if col not in keep_variables])


def drop_missing_address(df):
    return df.dropna(subset=["ADDRESS"])


def clean_edsa_address(text):
    # Standardize directional tags to (NB) and (SB)
    text = re.sub(r"(?i)\(?\bN\s*[./]?\s*B\b\.?\)?", "(NB)", text)
    text = re.sub(r"(?i)\(?\bS\s*[./]?\s*B\b\.?\)?", "(SB)", text)

    # Standardize Barangay variations to 'Brgy. '
    text = re.sub(r"(?i)\b(?:Bgy|Brgy|Barangay)\b\.?\s*", "Brgy. ", text)

    # Standardize U-Turn variations
    text = re.sub(r'(?i)"?u"?[- ]?turn', "U-Turn", text)

    # Standardize road type abbreviations
    text = re.sub(r"(?i)\bAve\.?(?=\s|$)", "Avenue", text)
    text = re.sub(r"(?i)\bSt\.?(?=\s|$)", "Street", text)
    text = re.sub(r"(?i)\bBlvd\.?(?=\s|$)", "Boulevard", text)

    # Expand city abbreviations
    text = re.sub(r"(?i)\bQ\.?C\.?\b", "Quezon City", text)
    text = re.sub(r"(?i)\bMkt\.?\b", "Makati", text)

    # Fix commas without trailing space
    text = re.sub(r",([^\s])", r", \1", text)

    # Collapse multiple internal spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def normalize_address_column(df):
    df = df.copy()
    df["ADDRESS"] = df["ADDRESS"].str.strip().str.title().apply(clean_edsa_address)
    return df


def extract_hour(df):
    df = df.copy()
    df["DATETIME_PST"] = pd.to_datetime(df["DATETIME_PST"], format="mixed")
    df["HOUR"] = df["DATETIME_PST"].dt.hour
    df = df.drop(columns=["DATETIME_PST"])
    return df


def find_invalid_range_rows(df, column, min_value, max_value):
    return df[(df[column] < min_value) | (df[column] > max_value)]


def validate_severity_values(df):
    return sorted(df["SEVERITY"].dropna().unique().tolist())


def validate_x_bounds(df):
    invalid = df[(df["X"] < X_MIN) | (df["X"] < X_MAX)]
    return len(invalid)


def validate_y_bounds(df):
    invalid = df[(df["Y"] < Y_MIN) | (df["Y"] > Y_MAX)]
    return len(invalid)


def validate_collision_type_values(df):
    invalid = df[df["COLLISION_TYPE"].isna() | (df["COLLISION_TYPE"].str.strip() == "")]
    return len(invalid)


def add_severity_num(df):
    df = df.copy()
    df["SEVERITY_NUM"] = df["SEVERITY"].map(SEVERITY_MAP)
    return df


def add_hour_bin(df):
    df = df.copy()
    df["HOUR_BIN"] = pd.cut(df["HOUR"], bins=HOUR_BIN_EDGES, labels=HOUR_BIN_LABELS, right=False)
    return df
