import numpy as np
import pandas as pd
import pytest

from models.data_processing import (
    DataPipeline,
    UNKNOWN_COLLISION_TYPE,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)


@pytest.fixture
def dp():
    return DataPipeline()


def valid_df():
    return pd.DataFrame({
        'DATETIME_PST': ['2016-01-01 08:00:00', '2016-01-01 14:00:00', '2016-01-01 20:00:00'],
        'SEVERITY': ['Property', 'Injury', 'Fatal'],
        'X': [121.0, 121.02, 121.01],
        'Y': [14.6, 14.61, 14.59],
        'COLLISION_TYPE': ['Rear-End', UNKNOWN_COLLISION_TYPE, 'Side Swipe'],
        'EXTRA_COL': [1, 2, 3],
    })


# EDSA-UT001
def test_select_study_variables_keeps_only_expected_columns(dp):
    result = dp.select_study_variables(valid_df())
    assert 'EXTRA_COL' not in result.columns
    assert list(result.columns) == ['DATETIME_PST', 'SEVERITY', 'X', 'Y', 'COLLISION_TYPE']


# EDSA-UT002
def test_select_study_variables_raises_on_missing_column(dp):
    df = valid_df().drop(columns=['X'])
    with pytest.raises(KeyError):
        dp.select_study_variables(df)


# EDSA-UT003
def test_audit_study_data_summarizes_dataset(dp):
    df = valid_df()[['DATETIME_PST', 'SEVERITY', 'X', 'Y', 'COLLISION_TYPE']]
    audit = dp.audit_study_data(df)
    assert audit['rows'] == 3
    assert audit['columns'] == 5
    assert audit['missing'].sum() == 0
    assert audit['exact_duplicate_rows'] == 0
    assert audit['severity_values'] == ['Fatal', 'Injury', 'Property']
    assert audit['collision_type_values'] == [UNKNOWN_COLLISION_TYPE, 'Rear-End', 'Side Swipe']


# EDSA-UT004
def test_find_apparent_duplicates_flags_repeated_study_rows(dp):
    df = pd.DataFrame({
        'DATETIME_PST': ['2016-01-01 08:00:00'] * 2 + ['2016-01-01 09:00:00'],
        'SEVERITY': ['Property'] * 2 + ['Injury'],
        'X': [121.0] * 2 + [121.02],
        'Y': [14.6] * 2 + [14.61],
        'COLLISION_TYPE': ['Rear-End'] * 2 + ['Side Swipe'],
    })
    result = dp.find_apparent_duplicates(df)
    assert len(result) == 2


# EDSA-UT005
def test_parse_datetime_features_derives_hour(dp):
    result = dp.parse_datetime_features(valid_df())
    assert list(result['HOUR']) == [8, 14, 20]
    assert pd.api.types.is_datetime64_any_dtype(result['DATETIME_PST'])


# EDSA-UT006
def test_find_invalid_coordinate_rows_flags_out_of_bounds_and_null(dp):
    df = pd.DataFrame({
        'X': [X_MIN, X_MAX + 1, np.nan],
        'Y': [Y_MIN, Y_MAX, Y_MAX],
    })
    result = dp.find_invalid_coordinate_rows(df)
    assert len(result) == 2


# EDSA-UT007
def test_filter_known_collision_types_removes_unknown_label(dp):
    result = dp.filter_known_collision_types(valid_df())
    assert len(result) == 2
    assert UNKNOWN_COLLISION_TYPE not in result['COLLISION_TYPE'].values


# EDSA-UT008
def test_validate_categories_passes_on_valid_data(dp):
    dp.validate_categories(valid_df())  # should not raise


# EDSA-UT009
def test_validate_categories_raises_on_unexpected_severity(dp):
    df = valid_df()
    df.loc[0, 'SEVERITY'] = 'Unknown'
    with pytest.raises(ValueError, match='severity'):
        dp.validate_categories(df)


# EDSA-UT010
def test_validate_categories_raises_on_unexpected_collision_type(dp):
    df = valid_df()
    df.loc[0, 'COLLISION_TYPE'] = 'Meteor Strike'
    with pytest.raises(ValueError, match='collision-type'):
        dp.validate_categories(df)


# EDSA-UT011
def test_hour_converstion_derives_expected_sin_cos(dp):
    df = pd.DataFrame({'HOUR': [0, 6, 12, 18]})
    result = dp.hour_converstion(df)
    np.testing.assert_allclose(result['HOUR_SIN'], [0.0, 1.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(result['HOUR_COS'], [1.0, 0.0, -1.0, 0.0], atol=1e-9)


# EDSA-UT012
def test_process_data_cleans_and_filters_end_to_end(dp):
    result = dp.process_data(valid_df())
    assert len(result) == 2
    assert 'EXTRA_COL' not in result.columns
    assert 'HOUR' in result.columns
    assert {'HOUR_SIN', 'HOUR_COS'}.issubset(result.columns)
    assert list(result.index) == [0, 1]


# EDSA-UT013
def test_process_data_raises_on_missing_values(dp):
    df = valid_df()
    df.loc[0, 'X'] = np.nan
    with pytest.raises(ValueError, match='missing values'):
        dp.process_data(df)


# EDSA-UT014
def test_process_data_raises_on_out_of_bounds_coordinates(dp):
    df = valid_df()
    df.loc[0, 'X'] = 999.0
    with pytest.raises(ValueError, match='outside EDSA bounds'):
        dp.process_data(df)


# EDSA-UT015
def test_onehot_encoding_creates_dummy_columns(dp):
    df = pd.DataFrame({
        'COLLISION_TYPE': ['Rear-End', 'Side Swipe'],
        'SEVERITY': ['Property', 'Injury'],
    })
    result = dp.Onehot_Encoding(df)
    assert 'COLLISION_TYPE_Rear-End' in result.columns
    assert 'SEVERITY_Injury' in result.columns
    assert result['COLLISION_TYPE_Rear-End'].tolist() == [1.0, 0.0]


# EDSA-UT016
def test_load_dataset_reads_csv(dp, tmp_path):
    csv_path = tmp_path / 'sample.csv'
    csv_path.write_text('X,Y\n121.0,14.6\n121.01,14.61\n')
    result = dp.load_dataset(csv_path)
    assert list(result.columns) == ['X', 'Y']
    assert len(result) == 2


# EDSA-UT017
def test_find_dataset_path_uses_given_start(dp, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    csv_path = dataset_dir / 'RTA_EDSA_2007-2016.csv'
    csv_path.write_text('X,Y\n121.0,14.6\n')

    found = dp.find_dataset_path(start=tmp_path)
    assert found == csv_path


# EDSA-UT018
def test_find_dataset_path_raises_when_not_found(dp, tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.find_dataset_path(start=tmp_path)


# --- CBDATSI Phase 2 additions -------------------------------------------------


# EDSA-UT046
def test_parse_datetime_features_omits_time_bin_by_default(dp):
    result = dp.parse_datetime_features(valid_df())
    assert 'TIME_BIN' not in result.columns


# EDSA-UT047
def test_parse_datetime_features_adds_time_bin_when_requested(dp):
    df = pd.DataFrame({
        'DATETIME_PST': [
            '2016-01-01 03:00:00', '2016-01-01 09:00:00',
            '2016-01-01 15:00:00', '2016-01-01 21:00:00',
        ],
    })
    result = dp.parse_datetime_features(df, include_time_bin=True)
    assert list(result['TIME_BIN'].astype(str)) == [
        'Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)',
    ]


# EDSA-UT048
def test_prepare_phase1_data_derives_time_bin_and_drops_unknown_collisions(dp):
    result = dp.prepare_phase1_data(valid_df())
    assert len(result) == 2
    assert 'EXTRA_COL' not in result.columns
    assert {'HOUR', 'TIME_BIN'}.issubset(result.columns)
    # Phase 1 keeps the raw hour only; the cyclic features belong to the CBADVAI pipeline.
    assert 'HOUR_SIN' not in result.columns
    assert list(result.index) == [0, 1]
