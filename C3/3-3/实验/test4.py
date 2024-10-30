import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


def test_DecisionTreeClassifier_deepth(*data, maxdeepth):
    X_train, X_test, y_train, y_test = data
    deepths = np.arange(1, maxdeepth)
    training_scores = []
    testing_scores = []
    for deepth in deepths:
        clf = DecisionTreeClassifier(max_depth=deepth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(deepths, training_scores, label="training score", marker='o')
    ax.plot(deepths, testing_scores, label="testing score", marker='x')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5, loc='best')
    plt.show()

        

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
test_DecisionTreeClassifier_deepth(X_train, X_test, y_train, y_test, maxdeepth=100)
