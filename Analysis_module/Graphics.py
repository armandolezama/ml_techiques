from sklearn.tree import export_graphviz, plot_tree
from graphviz import Source
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

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
        svm,
        X,
        response_method=response_method,
        cmap=plt.cm.Spectral,
        alpha=alpha,
        xlabel=xlabel,
        ylabel=ylabel,
    )


