"""
This python module contains useful functions to perform the EDA of the project
"""
import os
import sys
import yaml
import matplotlib.pyplot as plt

# Import yaml file in order to access global variables and paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import graph_properties

# Loading config file
config_name = "config.yaml"
with open(os.path.join(parent_dir, config_name), encoding="utf-8") as conf:
        config_f = yaml.safe_load(conf)

def graph_binary(df,variable, title, xticks):
    frequencies = df[variable].value_counts()
    fig, ax = plt.subplots()

    # Crear las barras
    barras = ax.bar(frequencies.index, frequencies.values, color=[config_f["colors"]["teal"], config_f["colors"]["rojo"]])
    for bar in barras:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(xticks)

    plt.show()