"""
Module that integrates the fitting process of a logistic regression, random forest classifier and an xgboost model
"""
import os
import sys
import yaml
import logging
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


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

from utils import get_predictions_and_probabilities, create_results_dataframe, prepare_data_for_modeling

# Configure logfile

log_filename = config_f["logging"]["logistic_regression"]
logging.basicConfig(filename=log_filename, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
##################### Data ############################################################################

X_train, X_test, y_train, y_test, categorical, numerical = prepare_data_for_modeling()

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
preprocessed_X_train = preprocessor_logistic_regression.fit_transform(X_train)
preprocessed_X_test = preprocessor_logistic_regression.transform(X_test)

pipeline = make_pipeline(
    preprocessor_logistic_regression,
    SMOTE(random_state=42),
    LogisticRegression(max_iter=2500, random_state=42)
)

# Parameters for grid search
param_grid = {
    'logisticregression__C': [0.3, 0.4, 0.5],
    'logisticregression__solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

## Fitting model
best_model = grid_search.best_estimator_
parameters_best_model = grid_search.best_params_

train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

## Save model and parameters to disk for replicability
filename_bm_params = config_f["models"]["logistic_regression"]["params"]
filename_bm = config_f["models"]["logistic_regression"]["model"]

logging.info(f"Training score \n logistic regression: {train_score:.4f}")
logging.info(f"Test score\n logistic regression: {test_score:.4f}")
logging.info(f"Parameters\n logistic regression: {parameters_best_model}")

try:
    with open(filename_bm_params, 'wb') as file_params:
        pickle.dump(parameters_best_model, file_params)
        logging.info(f"Parameters saved in: {filename_bm_params}")

    with open(filename_bm, 'wb') as file_bm:
        pickle.dump(best_model, file_bm)
        logging.info(f"Model saved in: {filename_bm}")

except Exception as e:
    logging.error(f"Error in saving model and parameters: {e}")
    sys.exit(1)

y_train_pred, y_train_prob = get_predictions_and_probabilities(best_model, X_train)
y_test_pred, y_test_prob = get_predictions_and_probabilities(best_model, X_test)


train_results = create_results_dataframe(X_train, y_train, y_train_pred, y_train_prob)
test_results = create_results_dataframe(X_test, y_test, y_test_pred, y_test_prob)

try:
    train_results_path = os.path.join(config_f['data']['results']['logistic_regression'], 'train_results.parquet')
    test_results_path = os.path.join(config_f['data']['results']['logistic_regression'], 'test_results.parquet')
    train_results.to_parquet(train_results_path, engine="pyarrow", compression=None)
    test_results.to_parquet(test_results_path, engine="pyarrow", compression=None)
    logging.info(f"Train results saved in: {train_results_path}")
    logging.info(f"Test results saved in: {test_results_path}")
except Exception as e:
    logging.error(f"Error saving results: {e}")
    sys.exit(1)
