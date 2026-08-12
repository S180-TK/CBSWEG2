# Project Name: EDSA Traffic Model: Predictions, Insights, & Classifications

## Project Overview
> This project conducts an exploratory data analysis (EDA) and develops a classification model on traffic data gathered by the MMDA around EDSA. The objective is to uncover underlying patterns, correlations, and trends to provide a clearer picture of EDSA's traffic and incident dynamics.

## Team Members
| Name                | Role                           | Responsibilities                                                                                                                                                                                                                                 |
|---------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ed Bennett Borromeo | Project Manager / Scrum Master | - Facilitates sprint planning, reviews, and retrospectives <br> - Maintains the Jira backlog and tracks task progress <br> - Ensures the team meets deadlines and resolves blockers <br> - Acts as the main point of contact for the instructor  |
| Rovick Dompor       | QA / Tester                    | - Designs and executes test cases for each feature <br> - Maintains the GitHub Actions automated test scripts <br> - Reports and tracks bugs via Jira issues <br> - Verifies fixes before tasks are marked as done                               |
| Matthew Fuentes     | Full Stack Developer           | - Develops and maintains both frontend and backend components <br> - Reviews and merges pull requests on GitHub <br> - Follows team coding conventions and file naming standards <br> - Writes and updates technical documentation               |

## Directory Structure
```
project-repo/
├── .github/
│   └── workflows/
│       └── test.yml
├── dataset/
├── src/
│   ├── models/
│   │   ├── data_processing.py     # shared cleaning pipeline (DataPipeline)
│   │   ├── eda.py                 # CBDATSI descriptive statistics
│   │   ├── eda_plots.py           # CBDATSI EDA visualizations
│   │   ├── clustering.py          # CBDATSI Phase 2 K-means pipeline
│   │   ├── cluster_plots.py       # CBDATSI Phase 2 cluster visualizations
│   │   ├── inference.py           # CBDATSI Phase 2 chi-square test
│   │   ├── logistic_regression.py # CBADVAI multinomial logistic regression
│   │   ├── mlp.py                 # CBADVAI multi-layer perceptron
│   │   ├── training.py            # CBADVAI K-fold tuning and final training
│   │   ├── evaluation.py          # CBADVAI classification metrics
│   │   └── plotting.py            # CBADVAI model visualizations
│   ├── notebooks/
│   │   ├── CBADVAI_MCO_V13.ipynb  # CBADVAI classification models
│   │   └── Phase_2-Final.ipynb    # CBDATSI Phases 1 and 2
│   └── tests/
├── conftest.py
├── requirements.txt
└── README.md
```
