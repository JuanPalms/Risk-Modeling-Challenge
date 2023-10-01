# Risk Modeling Challenge

This project contains the code for the risk modeling challenge. It is developed in Python, and a Conda environment is provided to replicate the project. Additionally, the project is delivered with a Docker container.

## Structure

```bash 
├── README.md
├── config.yaml # YAML file that contains directories, variables, and other global settings used throughout the project.
├── environment.yaml # Contains the Conda environment for the project.
├── graph_properties.py # Contains graph properties that are used throughout the project for consistency.
├── data
│   ├── clean
│   │   └── datos.parquet
│   └── raw
│       ├── credit_reports.parquet
│       └── main_dataset.parquet
├── logs
│   └── training_logs.log
├── models # Pickle files with the binary files of the models.
│   ├── logistic_regression
│   │   ├── logistic_regression_model.pkl
│   │   └── logistic_regression_params.pkl
│   ├── random_forest
│   └── xg_boost
├── notebooks  # Data exploration and results.
│   ├── EDA.ipynb # Walkthrough EDA using the custom function provided in the src/eda.py.
│   └── Modeling.ipynb # Walkthrough validation metrics for the trained models.
├── results
│   └── graphs
└── src
    ├── cleaning.py # Python module that processes the raw data and stores it in a clean data folder.
    ├── eda.py # Custom functions for the EDA.
    ├── tables.py # Custom function for producing an HTML-like table to display table results.
    └── training.py # Models training pipelines.
```