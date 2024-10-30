import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


def test_DecisionTreeClassifier(* data):
    X_train, X_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("Training score:%f" % (clf.score(X_train, y_train)))
    print("Testing score:%f" % (clf.score(X_test, y_test)))


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
