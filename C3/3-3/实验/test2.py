import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


def test_DecisionTreeClassifier_criterion(* data):
    X_train, X_test, y_train, y_test = data
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print("criterion:%s" % criterion)
        print("Training score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f" % (clf.score(X_test, y_test)))
        

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
test_DecisionTreeClassifier_criterion(X_train, X_test, y_train, y_test)
