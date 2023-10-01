"""
Module that integrates the fitting process of a logistic regression, random forest classifier and an xgboost model
"""
import os
import sys
import yaml
import pickle
import pandas as pd
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

###
from sklearn.metrics import accuracy_score

# Import yaml file in order to access global variables and paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Loading config file
config_name = "config.yaml"
with open(os.path.join(parent_dir, config_name), encoding="utf-8") as conf:
        config_f = yaml.safe_load(conf)

##################### Data ############################################################################

data = pd.read_parquet(os.path.join(config_f["data"]["data_clean"], "datos.parquet"), engine="pyarrow")
# Define X (independent variables) and y (target variable)
y = data["target"]
# Drop target variable from df to be processed
X = data.drop(columns=["target"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

categorical = ["LOAN_TERM_MONTHLY",'APPLICATION_MONTH', 'APPLICATION_YEAR']
numerical = ['BALANCE_DUE', 'CREDIT_LIMIT', 'FINANCED_AMOUNT',
             'account_to_application_days', 'n_sf_apps',
             'n_bnpl_apps','n_inquiries_l3m' ]




######################## Logistic regression #########################################################
# Preprocessing steps
#1) Transformes for numeric attributes
numeric_transformer = Pipeline(
    steps=[ ('impute', SimpleImputer(strategy='mean')),
            ("scaler", StandardScaler())]
    )

#2) transformers for nominal attributes
nominal_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

#3) Complete preprocessing steps
preprocessor_logistic_regression = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical),
        ("nom", nominal_transformer, categorical)
        ],
    remainder='drop'
    )

# Logistic regression pipeline
logistic_regression_pipeline = Pipeline(
    steps=[('preprocessor', preprocessor_logistic_regression),
           ('classifier', LogisticRegression(max_iter=2500, random_state=42))])

param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga']
}

grid_search_lr = GridSearchCV(logistic_regression_pipeline, param_grid, cv=5)
grid_search_lr.fit(X_train, y_train)

### Best logistic regression model and parameters to pickle files: 
best_model = grid_search_lr.best_estimator_
parameters_best_model=grid_search_lr.best_params_
filename_bm_params = config_f["models"]["logistic_regression"]["params"]
filename_bm= config_f["models"]["logistic_regression"]["model"]


with open(filename_bm_params, 'wb') as file_params:
    pickle.dump(parameters_best_model, file_params)


with open(filename_bm, 'wb') as file_bm:
    pickle.dump(best_model, file_bm)
