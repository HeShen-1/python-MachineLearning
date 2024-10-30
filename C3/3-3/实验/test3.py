import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


def test_DecisionTreeClassifier_splitter(* data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train, y_train)
        print("splitter:%s" % splitter)
        print("Training score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f" % (clf.score(X_test, y_test)))
        

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
test_DecisionTreeClassifier_splitter(X_train, X_test, y_train, y_test)