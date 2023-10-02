"""
This python module contains useful functions to perform the modeling of the project
"""

import os
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, recall_score, accuracy_score, ConfusionMatrixDisplay

# Import yaml file in order to access global variables and paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Loading config file
config_name = "config.yaml"
with open(os.path.join(parent_dir, config_name), encoding="utf-8") as conf:
        config_f = yaml.safe_load(conf)


def get_metrics(model):
    """
    Compute various evaluation metrics for a given model based on its predictions.

    Parameters:
    - model (str): The name of the model, used to retrieve the predictions.

    Returns:
    - metrics_df (pd.DataFrame): A dataframe containing metrics for both the training and test data.
    - cm_dict (dict): Dictionary containing confusion matrices for both training and test data.
    """
    
    # Function to compute metrics for a given dataset
    def compute_metrics(results):
        cm = confusion_matrix(results['actual_target'], results['predicted_target'])
        metrics = {
            'AUC': roc_auc_score(results['actual_target'], results['predicted_probability']),
            'F1': f1_score(results['actual_target'], results['predicted_target']),
            'Recall': recall_score(results['actual_target'], results['predicted_target']),
            'Accuracy': accuracy_score(results['actual_target'], results['predicted_target']),
            'FPR': cm[0,1] / (cm[0,1] + cm[0,0]), # False Positive Rate
            'FNR': cm[1,0] / (cm[1,0] + cm[1,1])  # False Negative Rate
        }
        return metrics, cm

    datasets = {
        "Train": pd.read_parquet(os.path.join(config_f['data']['results'][f'{model}'], 'train_results.parquet'), engine="pyarrow"),
        "Test": pd.read_parquet(os.path.join(config_f['data']['results'][f'{model}'], 'test_results.parquet'), engine="pyarrow")
    }

    metrics_data = {}
    cm_dict = {}
    for key, value in datasets.items():
        metrics, cm = compute_metrics(value)
        metrics_data[key] = metrics
        cm_dict[key] = cm

    metrics_df = pd.DataFrame(metrics_data).transpose()
    metrics_df['Set'] = metrics_df.index

    return metrics_df, cm_dict

def plot_matriz(matriz, labels, title):
    """
    Plot a confusion matrix with specified colors and save the plot to a designated location.

    Parameters:
    - matriz (np.array or list of lists): The confusion matrix to be plotted.
    - labels (list): List of unique class labels.
    - title (str): Title for the plot and also used as the filename for saving the plot.

    Returns:
    - None. The plot is displayed and saved to a designated location.
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = mcolors.ListedColormap([config_f['colors']['gris'], config_f['colors']['teal']])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=labels)
    
    color_matrix = np.zeros_like(matriz, dtype=int)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if i == j:
                color_matrix[i, j] = 0
            else:
                color_matrix[i, j] = 1
                
    disp.plot(cmap=cmap, ax=ax, values_format=".0f")
    ax.imshow(color_matrix, cmap=cmap, alpha=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Set text properties
    for text in ax.texts:
        text.set_color("black")
        text.set_weight('bold')
    
    plt.savefig(os.path.join(config_f['results']['graphs'], f'{title}.png'), bbox_inches='tight', pad_inches=0)
    plt.show()
