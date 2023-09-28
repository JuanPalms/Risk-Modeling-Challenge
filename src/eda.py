"""
This python module contains useful functions to perform the EDA of the project
"""
import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Import yaml file in order to access global variables and paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import graph_properties

# Loading config file
config_name = "config.yaml"
with open(os.path.join(parent_dir, config_name), encoding="utf-8") as conf:
        config_f = yaml.safe_load(conf)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def graph_categorical(df, variable, title, xticks=None, figsize=(10, 6)):
    """
    Plot a bar chart for a categorical variable.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame containing the data.
    - variable (str): The column name of the categorical variable to be plotted.
    - title (str): The title to display on the bar chart.
    - xticks (list, optional): A list of tick positions for the x-axis. If None, category names will be used. Default is None.
    - figsize (tuple, optional): A tuple (width, height) in inches defining the figure size. Default is (10, 6).

    Returns:
    None. Displays a bar chart.

    Notes:
    - The function uses colors defined in a global configuration (`config_f`).
    - Each bar will display its frequency value at the top.
    """

    frequencies = df[variable].value_counts().sort_index()  # Ordenar por índice
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars
    barras = ax.bar(frequencies.index, frequencies.values, color=[config_f["colors"]["teal"], config_f["colors"]["rojo"]])
    for bar in barras:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontweight="bold")
    
    # Set x-axis ticks
    if xticks is None:
        ax.set_xticks(np.arange(1, len(frequencies.index)+1))  # Centrar los xticks con las barras
        ax.set_xticklabels(frequencies.index)
    else:
        ax.set_xticks(xticks)

    plt.show()

    




def plot_panel(df, variables, nrows, ncols, plot_type="hist", y_var=None, figsize=(15, 10)):
    """
    Plots histograms, scatter plots, or box plots for the specified variables in a panel.

    Args:
    - df (pd.DataFrame): DataFrame containing the data to be plotted.
    - variables (list of str): List of variables/columns in the DataFrame to be plotted.
    - nrows (int): Number of rows for the panel.
    - ncols (int): Number of columns for the panel.
    - plot_type (str): Type of plot, either "hist" for histogram, "scatter" for scatter plot, or "box" for box plot.
    - y_var (str): If plot_type is "scatter" or "box", the variable to plot against.
    """

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, var in zip(axes.ravel(), variables):
        if plot_type == "hist":
            df[var].hist(ax=ax, color=config_f["colors"]["teal"], alpha=0.7)
            ax.set_title(var)
        elif plot_type == "scatter" and y_var:
            ax.scatter(df[y_var], df[var], color=config_f["colors"]["teal"], alpha=0.7)
            ax.set_title(f"{var} vs {y_var}")
            ax.set_xlabel(y_var)
            ax.set_ylabel(var)
        elif plot_type == "box" and y_var:
            df.boxplot(column=var, by=y_var, ax=ax, grid=False)
            ax.set_title(f"{var} by {y_var}")
            ax.set_xlabel(y_var)
            ax.set_ylabel(var)
        ax.grid(False)

    # Hide the empty subplots
    for ax in axes.ravel()[len(variables):]:
        ax.axis('off')

    # Adjust the layout to make space for the titles
    plt.tight_layout()

    # Remove suptitle added by boxplot
    if plot_type == "box":
        plt.suptitle('') 
    plt.show()


    
def cap_at_95th_percentile(df, column_name):
    """
    Caps the values of a specified column at the 95th percentile.
    
    This function takes a DataFrame and a column name, then limits 
    (caps) the values of that column to the 95th percentile.

    Args:
    - df (pd.DataFrame): The original DataFrame.
    - column_name (str): Name of the column to be capped.

    Returns:
    - pd.DataFrame: DataFrame with the specified column capped at the 95th percentile.
    """
    
    # Calculate the 95th percentile for the specified column
    percentile_95 = df[column_name].quantile(0.95)
    
    # Cap the values in the column at the 95th percentile
    df[column_name] = df[column_name].where(df[column_name] <= percentile_95, percentile_95)
    
    return df
