"""
This python module produces clean files for the model training process
"""
import os
import sys
import yaml
import pyarrow
import numpy as np
import pandas as pd

def load_config(config_path, config_name="config.yaml"):
    """
    Load the configuration from a YAML file.
    
    Parameters:
    - config_path (str): Path to the directory containing the configuration file.
    - config_name (str, optional): Name of the configuration file. Defaults to "config.yaml".
    
    Returns:
    - dict: Configuration dictionary loaded from the file.
    """
    with open(os.path.join(config_path, config_name), encoding="utf-8") as conf:
        return yaml.safe_load(conf)

def cap_at_95th_percentile(df, column):
    """
    Cap the values of a DataFrame column at its 95th percentile.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Name of the column to be capped.
    
    Returns:
    - pd.Series: Series with values capped at the 95th percentile of the original column.
    """
    upper = df[column].quantile(0.95)
    return df[column].clip(upper=upper)


categoricas = ['LOAN_TERM_MONTHLY']
numericas = ["FINANCED_AMOUNT", "account_to_application_days", "n_sf_apps", "n_bnpl_apps", "n_inquiries_l3m"]
credit_variables = ['BALANCE_DUE', 'CREDIT_LIMIT']
target = ["target"]

columns_keep = categoricas + numericas + credit_variables + target

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
config_f = load_config(parent_dir)

# Paths and file loading
main_path = os.path.join(config_f["data"]["data_raw"], "main_dataset.parquet")
credit_reports_path = os.path.join(config_f["data"]["data_raw"], "credit_reports.parquet")
main = pd.read_parquet(main_path, engine='pyarrow')
credit_reports = pd.read_parquet(credit_reports_path, engine='pyarrow')

df_total = (
    credit_reports[["LOAN_ID", "CUSTOMER_ID"] + credit_variables]
    .groupby(by=["LOAN_ID", "CUSTOMER_ID"])
    .mean()
    .reset_index()
    .merge(main, on=["LOAN_ID", "CUSTOMER_ID"], how="right")
    .fillna({col: 0 for col in numericas})
    .assign(
        APPLICATION_MONTH=lambda df: df["APPLICATION_DATETIME"].dt.month,
        APPLICATION_YEAR=lambda df: df["APPLICATION_DATETIME"].dt.year
    )
    .loc[:, lambda df: df.columns.isin(columns_keep + ['APPLICATION_MONTH', 'APPLICATION_YEAR'])]
)

# Cap credit variables at the 95th percentile
df_total[credit_variables] = df_total[credit_variables].apply(lambda col: cap_at_95th_percentile(df_total, col.name))

# produce final dataframe
df_total.to_parquet(os.path.join(config_f['data']['data_clean'],'datos.parquet'),engine="pyarrow", compression=None)

