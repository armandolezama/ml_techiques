import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC

class Analysis:
  def __init__(self) -> None:
    print('hi from analysis')

  def get_tree_instance(self, X, y, criterion, max_depth, use_regressor=False):
    tree_instance = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth) if(use_regressor) else  DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    tree_instance.fit(X, y)

    return tree_instance

  def get_svm_instance(self, X, y, kernel, gamma, C):
    svm = SVC(kernel=kernel, gamma=gamma, C=C)
    svm.fit(X, y)

    return svm

  def get_random_forest(self):
    print('hello')