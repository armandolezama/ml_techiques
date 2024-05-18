from sklearn.tree import export_graphviz, plot_tree
from graphviz import Source
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import seaborn as sns

class Graphics:
  def __init__(self) -> None:
    print('hi from graphics')

  def plot_tree(self, tree_instance, feature_names: list[str], target, file_name: str):
    if not isinstance(tree_instance, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError("tree_instance must be a DecisionTreeClassifier or DecisionTreeRegressor")

    export_graphviz(
        decision_tree=tree_instance,
        out_file=file_name,
        feature_names=feature_names,
        class_names=target,
        rounded=True,
        filled=True
    )

    with open(file_name) as f:
        dot_graph = f.read()

    Source(dot_graph)

  def plot_decision_boundary(self, svm:SVC, X, response_method, alpha, xlabel, ylabel):
    DecisionBoundaryDisplay.from_estimator(
        estimator=svm,
        X=X,
        response_method=response_method,
        cmap=plt.cm.Spectral,
        alpha=alpha,
        xlabel=xlabel,
        ylabel=ylabel,
    )

  def plot_numeric_variables(self, data: pd.DataFrame, plots_per_row: int = 5, plot_type: str = 'histogram', y: pd.Series = None):
    # Configuración de estilo
    plt.style.use('seaborn-darkgrid')
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Calcular el número de filas necesarias
    num_plots = len(numeric_columns)
    num_rows = (num_plots // plots_per_row) + int(num_plots % plots_per_row != 0)

    # Crear subplots
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(plots_per_row * 5, num_rows * 4), facecolor='white')
    axes = axes.flatten()

    # Crear gráficos según el tipo especificado
    for i, column in enumerate(numeric_columns):
        if plot_type == 'histogram':
            sns.histplot(data[column], kde=True, ax=axes[i])
            axes[i].set_title('Distribución de ' + column)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frecuencia')
        elif plot_type == 'scatter' and y is not None:
            sns.scatterplot(x=data[column], y=y, ax=axes[i])
            axes[i].set_title('Relación entre ' + column + ' y target')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('target')
        elif plot_type == 'box':
            sns.boxplot(y=data[column], ax=axes[i])
            axes[i].set_title('Caja de ' + column)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Valores')

    # Eliminar subplots vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

  def plot_scatter_plots(self, X: pd.DataFrame, y: pd.Series, plots_per_row: int = 5):
    # Configuración de estilo
    plt.style.use('seaborn-darkgrid')
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

    # Calcular el número de filas necesarias
    num_plots = len(numeric_columns)
    num_rows = (num_plots // plots_per_row) + int(num_plots % plots_per_row != 0)

    # Crear subplots
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(plots_per_row * 5, num_rows * 4), facecolor='white')
    axes = axes.flatten()

    # Crear gráficos de dispersión para cada variable numérica
    for i, column in enumerate(numeric_columns):
        sns.scatterplot(x=X[column], y=y, ax=axes[i])
        axes[i].set_title('Relación entre ' + column + ' y target')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('target')

    # Eliminar subplots vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

  def plot_correlation_matrix(self, data: pd.DataFrame):
    # Calcular la matriz de correlación
    correlation_matrix = data.corr()

    # Crear un mapa de calor para visualizar la matriz de correlación
    plt.figure(figsize=(12, 10), facecolor='white')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación')
    plt.show()

