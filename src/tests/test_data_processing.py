import pandas as pd

from models.data_processing import (
    add_hour_bin,
    add_severity_num,
    clean_edsa_address,
    drop_missing_address,
    extract_hour,
    filter_columns,
    find_invalid_range_rows,
    normalize_address_column,
)


def test_filter_columns_keeps_only_expected_variables():
    df = pd.DataFrame({
        "ADDRESS": ["A"], "DATETIME_PST": ["2016-01-01 01:00"], "SEVERITY": ["Fatal"],
        "X": [121.0], "Y": [14.6], "COLLISION_TYPE": ["Self-Accident"], "EXTRA_COL": [1],
    })
    result = filter_columns(df)
    assert "EXTRA_COL" not in result.columns
    assert list(result.columns) == ["ADDRESS", "DATETIME_PST", "SEVERITY", "X", "Y", "COLLISION_TYPE"]


def test_drop_missing_address_removes_nan_rows():
    df = pd.DataFrame({"ADDRESS": ["EDSA Guadalupe", None, "EDSA Cubao"]})
    result = drop_missing_address(df)
    assert result["ADDRESS"].isnull().sum() == 0
    assert len(result) == 2


def test_clean_edsa_address_standardizes_directions_and_barangay():
    assert clean_edsa_address("EDSA N.B. Bgy Guadalupe") == "EDSA (NB) Brgy. Guadalupe"
    assert clean_edsa_address("EDSA S/B Barangay Cubao") == "EDSA (SB) Brgy. Cubao"


def test_clean_edsa_address_expands_road_and_city_abbreviations():
    # NOTE: a trailing "." survives when the abbreviation ends the string -
    # this mirrors the existing regex behavior carried over from the notebook.
    assert clean_edsa_address("Ortigas Ave. Q.C.") == "Ortigas Avenue Quezon City."
    assert clean_edsa_address("Buendia St. Mkt.") == "Buendia Street Makati."


def test_normalize_address_column_strips_and_titles():
    df = pd.DataFrame({"ADDRESS": ["  edsa guadalupe  "]})
    result = normalize_address_column(df)
    assert result["ADDRESS"].iloc[0] == "Edsa Guadalupe"


def test_extract_hour_creates_hour_and_drops_datetime():
    df = pd.DataFrame({"DATETIME_PST": ["2016-05-01 14:30:00"]})
    result = extract_hour(df)
    assert "DATETIME_PST" not in result.columns
    assert result["HOUR"].iloc[0] == 14


def test_find_invalid_range_rows_flags_out_of_bounds_values():
    df = pd.DataFrame({"HOUR": [-1, 5, 23, 24]})
    result = find_invalid_range_rows(df, "HOUR", 0, 23)
    assert len(result) == 2
    assert set(result["HOUR"]) == {-1, 24}


def test_add_severity_num_maps_expected_ordinal_values():
    df = pd.DataFrame({"SEVERITY": ["Property", "Injury", "Fatal"]})
    result = add_severity_num(df)
    assert list(result["SEVERITY_NUM"]) == [1, 2, 3]


def test_add_hour_bin_assigns_expected_labels():
    df = pd.DataFrame({"HOUR": [0, 7, 13, 19]})
    result = add_hour_bin(df)
    assert list(result["HOUR_BIN"]) == [
        "Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)",
    ]
