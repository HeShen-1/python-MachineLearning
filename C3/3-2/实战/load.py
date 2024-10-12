from sklearn import datasets, linear_model, model_selection
import numpy as np
import pytest
import matplotlib.pyplot as plt


def load_diabetes():
    diabetes = datasets.load_diabetes()
    x = diabetes.data
    y = diabetes.target
    return model_selection.train_test_split(x, y, test_size=0.25, random_state=0)


def load_iris():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return model_selection.train_test_split(x, y, test_size=0.3, random_state=1, shuffle=True, stratify=y)


@pytest.fixture
def data1():
    """Fixture to load data."""
    return load_diabetes()


@pytest.fixture
def data2():
    """Fixture to load data."""
    return load_iris()


def test_LinearRegression(data1):
    X_train, X_test, y_train, y_test = data1
    regression = linear_model.LinearRegression()
    regression.fit(X_train, y_train)
    print('\nCoefficients:%s, intercept:%.2f' % (regression.coef_, regression.intercept_))
    print("Residual sum of squares:%.2f" % np.mean((regression.predict(X_test) - y_test) ** 2))
    print('Score:%.2f' % regression.score(X_test, y_test))


def test_LogisticRegression(data2):
    X_train, X_test, y_train, y_test = data2
    regression = linear_model.LogisticRegression()
    regression.fit(X_train, y_train)
    print('\nCoefficients:%s, intercept:%s' % (regression.coef_, regression.intercept_))
    print('Score:%.2f' % regression.score(X_test, y_test))


def test_LogisticRegression_multinomial(data2):
    X_train, X_test, y_train, y_test = data2
    regression = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regression.fit(X_train, y_train)
    print('\nCoefficients:%s, intercept:%s' % (regression.coef_, regression.intercept_))
    print('Score:%.2f' % regression.score(X_test, y_test))


def test_LogisticRegression_C(data2):
    X_train, X_test, y_train, y_test = data2
    Cs = np.logspace(-2, 4, num=100)
    scores = []
    for C in Cs:
        regression = linear_model.LogisticRegression(C=C)
        regression.fit(X_train, y_train)
        scores.append(regression.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title(r"Logistic Regression")
    plt.show()
