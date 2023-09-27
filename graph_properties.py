"""
This python module defines the global properties for the graphs in matplotlib
"""
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  # Importar FuncFormatter


# Establecer el tamaño predeterminado de la figura
plt.rcParams["figure.figsize"] = (8, 4)
# Establecer fondo transparente de la figura
plt.rcParams["figure.facecolor"] = "none"
# Eliminar los ejes superior y derecho
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
# Eliminar offset en ejes
plt.rcParams['axes.formatter.useoffset'] = False
# Evitar notación científica
plt.rcParams['axes.formatter.use_mathtext'] = False
# Tamaño de la letra para las etiquetas de los ejes
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Función para formatear con comas como separador de miles
def thousands(x, pos):
    return '{:,.0f}'.format(x)

def custom_format(x, pos):
    if x >= 1_000_000:
        return '{:.0f}M'.format(x * 1e-6)
    elif x >= 1_000:
        return '{:.1f}K'.format(x * 1e-3)
    else:
        return str(x)
