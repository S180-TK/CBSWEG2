# CBSWEG2 MCO2: Continuous Integration Plan

## Project Information

|                         |                                                                                                                                                                                                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Name**        | EDSA Traffic Model: Predictions, Insights, & Classifications                                                                                                                                                                                                                                   |
| **Project Description** | This project conducts an exploratory data analysis (EDA) and develops a classification model on traffic data gathered by the MMDA around EDSA. The objective is to uncover underlying patterns, correlations, and trends to provide a clearer picture of EDSA's traffic and incident dynamics. |
| **Team Members**        | _Ed Bennett Borromeo_ – Project Manager / Scrum Master<br>_Rovick Dompor_ – QA / Tester<br>_Matthew Fuentes_ – Full Stack Developer                                                                                                                                                            |
| **GitHub Repository**   | [GitHub - S180-TK/CBSWEG2](https://github.com/S180-TK/CBSWEG2)                                                                                                                                                                                                                                 |

## Technical Requirements

|                          | Name               | Version                                         | Description / Purpose                                                                                      |
| ------------------------ | ------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Programming Language** | Python             | 3.13.2                                          | Python contains a set of useful libraries for the completion of the requirements.                          |
| **Libraries**            | NumPy              | 2.4.0                                           | Arrays for data. Contains useful operations involving arrays.                                              |
|                          | Matplotlib         | 3.11.0                                          | Used for Data Visualization through graphs.                                                                |
|                          | pandas             | 3.0.3                                           | Pandas' DataFrame for use in the dataset. Used for Data Analysis.                                          |
| **Testing Framework**    | pytest             | 9.1.1                                           | Unit and integration testing; runs via GitHub Actions on push.                                             |
| **Operating System**     | Windows & macOS    | Windows 11 / macOS Tahoe 26.5.1                 | These two operating systems are what the team is using. It's essential so that all members can utilize it. |
| **Other Tools**          | Visual Code Studio | 1.125                                           | Coding editor for the project. Where the code will be created.                                             |
|                          | GitHub Actions     | v2.335.1                                        | Quick testing on edits to code from the Git repository itself.                                             |
|                          | Jupyter Notebook   | 7.6<br>VS Code Extension Ver: 2026.6.2026061001 | Interactive notebooks for EDA; CI/CD pipeline; project and issue tracking                                  |

## Repository Structure

| File / Folder        | Description                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `README.md`          | Contains info regarding the project, including: Project Name, Project Description, Team Members and Roles, Directory Structure |
| `requirements.txt`   | Python package dependencies for reproducible environment setup                                                                 |
| `.github/workflows/` | GitHub Actions workflow definitions (e.g. `test.yml`)                                                                          |
| `dataset/`           | Contains the raw dataset files (e.g. CSV) for the project.                                                                     |
| `src/`               | Main source directory containing notebooks, model scripts, and tests                                                           |
| `src/notebooks/`     | Jupyter notebooks for EDA, data preprocessing, and model development                                                          |
| `src/models/`        | Python scripts containing reusable and testable model and data processing functions                                            |
| `src/tests/`         | pytest test cases for data processing functions and model utilities                                                            |

## Branching Strategy

| Branch Name                | Description                                                                                                                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main`                     | Stable, production-ready branch. Only receives merges from `dev` via pull request after team review and all CI checks pass. No direct commits.                                                                          |
| `dev`                      | Integration branch where completed features are merged and tested together before promoting to `main`. CI runs automatically on every push to this branch.                                                              |
| `feature/<short-description>` | Short-lived branches for individual tasks (e.g. `feature/eda-cleaning`, `feature/classification-model`). Created from `dev`, merged back into `dev` via pull request. |

## Workflows

| Trigger        | Branch           | Tasks                                                                                                                                                                                                                                                                                  |
| -------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pull Request   | feature branches | 1. Checkout repository<br>2. Set up Python environment<br>3. Install dependencies (`requirements.txt`)<br>4. Run pytest test suite:<br>&nbsp;&nbsp;a. Unit Testing<br>&nbsp;&nbsp;b. Integration Testing<br>5. Report test results                                                     |
| Push           | `dev`, `main`    | 1. Checkout repository<br>2. Set up Python environment<br>3. Install dependencies (`requirements.txt`)<br>4. Run pytest test suite<br>5. Report test results                                                                                                                           |
| Manual Trigger | model branches   | 1. Checkout repository<br>2. Set up Python environment<br>3. Install dependencies (`requirements.txt`)<br>4. Run pytest suite<br>5. Load training dataset<br>6. Train machine learning model<br>7. Validate and evaluate model on test set<br>8. Output model metrics and test results |
