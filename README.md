# Risk Modeling Challenge

This repository hosts the code and resources for the risk modeling challenge.

## Solutions

Solutions can be found in the `notebooks` directory:

1. **`EDA.ipynb`**: This notebook includes the exploratory data analysis for the project. The main functions and scripts used here are located in the `src/EDA.py` script.
2. **`Modeling.ipynb`**: Here, the training process and evaluation metrics for the three proposed algorithms are showcased. The core code used in this notebook has been modularized and is stored in the `src` directory.

## Project Structure

```plaintext
Risk Modeling Project
├── README.md
├── config.yaml             # Global settings, variables, and directories for the project.
├── environment.yaml        # Conda environment specifications.
├── graph_properties.py     # Graph attributes for visual consistency throughout the project.
│
├── data
│   ├── clean
│   │   ├── datos.parquet
│   │   ├── datos_test.parquet
│   │   └── datos_train.parquet
│   ├── raw
│   │   ├── credit_reports.parquet
│   │   └── main_dataset.parquet
│   └── results
│       ├── logistic_regression
│       │   ├── test_results.parquet
│       │   └── train_results.parquet
│       ├── random_forest
│       │   ├── test_results.parquet
│       │   └── train_results.parquet
│       └── xgboost
│           ├── test_results.parquet
│           └── train_results.parquet
│
├── logs                     # Training logs for each algorithm.
│   ├── logistic_regression_logs.log
│   ├── random_forest_logs.log
│   └── xgboost_logs.log
│
├── models                  # Serialized models, preprocessors, and training parameters.
│   ├── logistic_regression
│   │   ├── logistic_regression_model.pkl
│   │   ├── logistic_regression_params.pkl
│   │   └── logistic_regression_preprocessor.pkl
│   ├── random_forest
│   │   ├── random_forest_model.pkl
│   │   ├── random_forest_params.pkl
│   │   └── random_forest_preprocessor.pkl
│   └── xgboost
│       ├── xgboost_model.pkl
│       ├── xgboost_params.pkl
│       └── xgboost_preprocessor.pkl
│
├── notebooks
│   ├── EDA.ipynb            # Exploratory data analysis.
│   └── Modeling.ipynb      # Detailed overview of preprocessing, training, and evaluation.
│
├── results
│   ├── graphs
│   │   ├── confusion_matrix_logistic_regression_test.png
│   │   ├── confusion_matrix_logistic_regression_train.png
│   │   ├── confusion_matrix_random_forest_test.png
│   │   ├── confusion_matrix_random_forest_train.png
│   │   ├── confusion_matrix_xgboost_test.png
│   │   └── confusion_matrix_xgboost_train.png
│   └── tables
│       ├── evaluation_metrics_overview.png
│       ├── evaluation_metrics_random_forest.png
│       └── evaluation_metrics_xgboost.png
│
└── src
    ├── cleaning.py           # Data cleaning pipeline.
    ├── eda.py                # EDA-specific functions.
    ├── logistic_regression.py
    ├── random_forest.py
    ├── xgboost_classifier.py
    ├── modeling.py           # Utility functions for model training and evaluation.
    └── utils.py

```

## Main Findings

For this project, I established a replicable pipeline for each proposed model and serialized the models for future use. While the outcomes weren't as expected due to the models achieving lower predictive power on client defaults, the pipelines serve as a solid foundation for refining and creating better models to estimate the probability of client defaults. Both training and test results indicated a bias in predictions. This can potentially be addressed by enhancing feature engineering and incorporating more predictors.

Below are some of the key results:
Logistic Regression:

![Confussion matrix train set](https://github.com/JuanPalms/Risk-Modeling-Challenge/blob/main/results/graphs/Confussion%20Matrix%0A%20Logistic%20regression%3A%20Test%20set.png)

![Evaluation metrics: Logistic regression](https://github.com/JuanPalms/Risk-Modeling-Challenge/blob/main/results/tables/Evaluation%20metrics.png)
