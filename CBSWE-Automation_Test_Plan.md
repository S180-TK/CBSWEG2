# Automation Test Plan

## CBSWEG2 MCO3: Automation Test Plan

## Project Information

| Field                   | Details                                                                                                                                                                                                                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Description** | This project conducts an exploratory data analysis (EDA) and develops a classification model on traffic data gathered by the MMDA around EDSA. The objective is to uncover underlying patterns, correlations, and trends to provide a clearer picture of EDSA's traffic and incident dynamics. |
| **Dataset**             | `RTA_EDSA_2007-2016.csv` — Road Traffic Accident Data of Epifanio delos Santos Avenue, Metro Manila (2007-2016), sourced from Mendeley Data.                                                                                                                                                   |
| **Team Members**        | Ed Bennett Borromeo – Project Manager / Scrum Master<br>Rovick Dompor – QA / Tester<br>Matthew Fuentes – Full Stack Developer                                                                                                                                                                  |
| **Project Name**        | EDSA Traffic Model: Predictions, Insights, & Classifications                                                                                                                                                                                                                                   |
| **GitHub Repository**   | [GitHub - S180-TK/CBSWEG2](https://github.com/S180-TK/CBSWEG2)                                                                                                                                                                                                                                 |

---

## Unit Testing

### Module: Data Cleaning / Preprocessing

_(`src/models/data_processing.py`)_

| ID          | Function                           | Test Description                                                                                                          | Input                                              | Expected Output                                                             |
| ----------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| EDSA-UT001  | `filter_columns()`                 | Keeps only the required variables (`ADDRESS`, `DATETIME_PST`, `SEVERITY`, `X`, `Y`, `COLLISION_TYPE`) and drops the rest. | DataFrame with those 6 columns plus an `EXTRA_COL` | DataFrame with `EXTRA_COL` removed; only the 6 kept columns remain          |
| EDSA-UT002  | `drop_missing_address()`           | Removes rows where `ADDRESS` is null.                                                                                     | `["EDSA Guadalupe", None, "EDSA Cubao"]`           | 2 rows remain; no nulls in `ADDRESS`                                        |
| EDSA-UT003  | `clean_edsa_address()`             | Standardizes N.B./S.B. directional tags and Barangay abbreviations.                                                       | `"EDSA N.B. Bgy Guadalupe"`                        | `"EDSA (NB) Brgy. Guadalupe"`                                               |
| EDSA-UT004  | `clean_edsa_address()`             | Expands road-type and city abbreviations (`Ave.`, `St.`, `Q.C.`, `Mkt.`).                                                 | `"Ortigas Ave. Q.C."`                              | `"Ortigas Avenue Quezon City."`                                             |
| EDSA-UT005  | `normalize_address_column()`       | Strips whitespace and titlecases address text.                                                                            | `" edsa guadalupe"`                                | `"Edsa Guadalupe"`                                                          |
| EDSA-UT006  | `extract_hour()`                   | Parses `DATETIME_PST`, derives an `HOUR` column, and drops `DATETIME_PST`.                                                | `"2016-05-01 14:30:00"`                            | `HOUR = 14`; `DATETIME_PST` column removed                                  |
| EDSA-UT007  | `find_invalid_range_rows()`        | Flags values outside a valid numeric range (e.g. `HOUR` outside 0–23).                                                    | `HOUR = [-1, 5, 23, 24]`, range (0, 23)            | Rows with `HOUR = -1` and `HOUR = 24` returned                              |
| EDSA-UT008  | `add_hour_bin()`                   | Bins `HOUR` into 4 time-of-day categories.                                                                                | `[0, 7, 13, 19]`                                   | `["Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"]` |
| EDSA-UT009  | `validate_severity_values()`       | Confirms the `SEVERITY` labels remain within known dataset categories.                                                    | `SEVERITY` column                                  | `["Property", "Injury", "Fatal"]`                                           |
| EDSA-UT010  | `validate_x_bounds()`              | Checks longitude values against the expected EDSA corridor range used in CBDATSI.                                         | `X` values                                         | All X values are within 120.96 to 121.08; invalid count = 0                 |
| EDSA-UT011  | `validate_y_bounds()`              | Checks latitude values against the expected EDSA corridor range used in CBDATSI.                                          | `Y` values                                         | All Y values are within 14.51 to 14.68; invalid count = 0                   |
| EDSA-UT012  | `validate_collision_type_values()` | Confirms the `COLLISION_TYPE` labels remain within known dataset categories.                                              | `COLLISION_TYPE` column                            | No blank or unexpected collision type labels are found                      |

> **Note:** Add a new module table (e.g. "Classification Model") with its own `EDSA-UT-0XX` rows once model-related functions exist.

---

## System Testing

End-to-end flow from raw dataset to final output, through the cleaning pipeline in `src/models/data_processing.py`:

```
RTA_EDSA_2007-2016.csv → Data Loading → Column Selection → Data Cleaning →
Feature Engineering → Validation → Preprocessing → Model Training/Inference →
Evaluation → Prediction/Evaluation Output
```

| Input                            | Module              | Core Function(s)                                                                      | Connects To         | System Test Role                                                             |
| -------------------------------- | ------------------- | ------------------------------------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------- |
| Dataset `RTA_EDSA_2007-2016.csv` | Data Loading        | `pandas.read_csv()`                                                                   | Column Selection    | Provides the MMDA EDSA accident records used by the pipeline.                |
| Data Loading                     | Column Selection    | `select_relevant_columns()`                                                           | Data Cleaning       | Loads the CSV into a DataFrame for processing.                               |
| Column Selection                 | Data Cleaning       | `drop_missing_address_rows()`, `clean_edsa_address()`                                 | Feature Engineering | Keeps `ADDRESS`, `DATETIME_PST`, `SEVERITY`, `X`, `Y`, and `COLLISION_TYPE`. |
| Data Cleaning                    | Feature Engineering | `parse_datetime_pst()`, `extract_hour()`                                              | Validation          | Removes missing addresses and standardizes address text.                     |
| Feature Engineering              | Validation          | `validate_hour_range()`, `validate_coordinate_bounds()`, `validate_category_values()` | Preprocessing       | Creates the `HOUR` feature from `DATETIME_PST`.                              |
| Validation                       | Preprocessing       | `split_features_target()`, `build_preprocessor()`, `transform_features()`             | Modeling            | Checks hours, EDSA coordinate bounds, and valid category labels.             |
| Preprocessing                    | Modeling            | `train_model()`, `predict_accident_outcome()`                                         | Evaluation / Output | Transforms cleaned fields into model-ready numeric/categorical features.     |
| Modeling                         | Evaluation / Output | `evaluate_model()`, `classification_report()`, `confusion_matrix()`                   | Output              | Trains the classifier and generates predicted accident outcome/risk class.   |
| Evaluation / Output              | Output              | —                                                                                     | —                   | Reports accuracy, precision, recall, F1-score, and confusion matrix.         |

---

## Performance Evaluation

> **Blocked** on the classification model, which hasn't been built yet (Phase 1 covered EDA/cleaning only). Once the model exists, these are the standard classification metrics — recommended given `SEVERITY` is heavily imbalanced (from the EDA: ~20,279 Property vs. 1,468 Injury vs. only 21 Fatal), so accuracy alone would be misleading.

| Metric                     | Formula                                         | Description                                                                             | Target Score                                                 |
| -------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Accuracy                   | (TP + TN) / Total predictions                   | Overall proportion of correct predictions across all classes.                           | ≥ 0.80 or higher than baseline majority-class model          |
| Precision (macro/weighted) | TP / (TP + FP)                                  | Measures how reliable positive predictions are for each class, averaged across classes. | ≥ 0.70, with attention to minority classes                   |
| Recall (macro/weighted)    | TP / (TP + FN)                                  | Measures how many actual class cases the model successfully identifies.                 | ≥ 0.70, especially for Injury/Fatal or casualty-risk classes |
| F1-score (macro/weighted)  | 2 _ (Precision _ Recall) / (Precision + Recall) | Balances precision and recall for imbalanced accident outcomes.                         | ≥ 0.70 and higher than baseline                              |
| Confusion Matrix           | Class-by-class count matrix                     | Shows which accident outcome classes are most often confused.                           | No severe class should be consistently missed                |
