"""
This python module defines the necesary functions to generate png tables of metrics
"""
import pandas as pd
import numpy as np
import os
import yaml
import sys
import matplotlib.pyplot as plt
import six
import pickle
import pyarrow
from sklearn.model_selection import train_test_split


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.options.display.float_format = '{:,.2f}'.format
# Loading config file
config_name = "config.yaml"
with open(os.path.join(parent_dir, config_name), encoding="utf-8") as conf:
        config_f = yaml.safe_load(conf)

def load_pickle(filename):
    """
    This function is used to load a pickle file.
    
    Parameters:
    -filename (str): Path to the binary file
    
    Returns:
    - sklearn serialized object
    """
    with open(filename, 'rb') as file:
            loaded_file = pickle.load(file)
    return loaded_file


def prepare_data_for_modeling():
    """
    Prepares data for modeling by loading, splitting and saving training/test datasets.
    
    Parameters:
    - model_name (str): Name of the model for logging purposes.
    
    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    try:
        data = pd.read_parquet(os.path.join(config_f["data"]["data_clean"], "datos.parquet"), engine="pyarrow")
    except Exception as e:
        logging.error(f"Error in loading data: {e}")
        sys.exit(1)

    y = data["target"]
    X = data.drop(columns=["target"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = pd.concat([X_train, y_train], axis=1).to_parquet(os.path.join(config_f['data']['data_clean'], 'datos_train.parquet'), engine="pyarrow", compression=None)
    test_df = pd.concat([X_test, y_test], axis=1).to_parquet(os.path.join(config_f['data']['data_clean'], 'datos_test.parquet'), engine="pyarrow", compression=None)
    
    categorical = config_f['variables']['categorical']
    numerical = config_f['variables']['numerical']
    
    return X_train, X_test, y_train, y_test, categorical, numerical


#### Useful functions for all the models

def get_predictions_and_probabilities(model, X):
    """
    Retrieve class predictions and their associated probabilities for the positive class.
    
    Parameters:
    - model (estimator): Trained scikit-learn model or similar with a predict() and predict_proba() methods.
    - X (pd.DataFrame or array-like): Dataset to predict on.
    
    Returns:
    - tuple: Predicted classes and associated probabilities.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities

def create_results_dataframe(X, actual_target, predicted_target, predicted_probability):
    """
    Construct a results dataframe appending actual targets, predicted targets, and predicted probabilities.
    
    Parameters:
    - X (pd.DataFrame): Original dataset without target column.
    - actual_target (array-like): True target values.
    - predicted_target (array-like): Predicted class labels.
    - predicted_probability (array-like): Predicted probabilities for the positive class.
    
    Returns:
    - pd.DataFrame: Extended dataframe with prediction results.
    """
    results_df = X.copy()
    results_df['actual_target'] = actual_target
    results_df['predicted_target'] = predicted_target
    results_df['predicted_probability'] = predicted_probability
    return results_df


        
def render_mpl_table(data, col_width=3.0, row_height=1, font_size=16,font_size_rows=16,
                     header_color=config_f['colors']['gris'], row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, title="Titulo", show_index=False, **kwargs):
    """
    This function is used to display a table using matplotlib. 
    It takes a pandas DataFrame and creates a plot with a table.

    Parameters:
    - data (pd.DataFrame): The data to display in the table.
    - col_width (float, optional): The relative width of each column. Default is 3.0.
    - row_height (float, optional): The relative height of each row. Default is 1.0.
    - font_size (int, optional): The font size to use. Default is 14.
    - header_color (str, optional): The color to use for the header row. Default is '#40466e'.
    - row_colors (list of str, optional): The colors to use for the non-header rows. Default is ['#f1f1f2', 'w'].
    - edge_color (str, optional): The cell border color. Default is 'w'.
    - bbox (list of float, optional): The bounding box in which the table is situated. Default is [0, 0, 1, 1].
    - header_columns (int, optional): The number of header columns. Default is 0.
    - ax (matplotlib.Axes, optional): An existing matplotlib Axes. If not provided, a new one is created.
    - **kwargs: Additional arguments passed to matplotlib.pyplot.table()

    Returns:
    - ax (matplotlib.Axes): A matplotlib Axes containing the rendered table.
    """

    
    if show_index:
        data = data.reset_index()

    data = data.applymap(lambda x: '{:,}'.format(round(x, 2)) if isinstance(x, (int, float)) else x)

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.set_fontsize(font_size_rows)
            cell.set_text_props(ha='center')
    plt.title(title, fontsize=20)
    plt.savefig(os.path.join(config_f['results']['tables'],
                                 f'{title}.png'), 
                    bbox_inches='tight', pad_inches=0)
    plt.show()
