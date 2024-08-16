from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class Analysis:
  def __init__(self) -> None:
    print('hi from analysis')

  def get_tree_instance(self, X, y, criterion, max_depth, use_regressor=False):
    tree_instance = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth) if(use_regressor) else  DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    tree_instance.fit(X, y)

    return tree_instance

  def support_vector_machine(self, X, y):
    model = SVC()
    model.fit(X, y)
    return model

  def perform_decision_tree(self, X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

  def perform_linear_regression(self, X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

  def logistic_regression(self, X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

  def neural_network(self, X, y):
    model = MLPClassifier()
    model.fit(X, y)
    return model

  def validate_model(self, model, X, y):
    return model.score(X, y)
