import numpy as np
import matplotlib.pyplot as plt
import pytest

from sklearn import neighbors
from sklearn.model_selection import train_test_split


def create_regression_data(n):
    """
    创建回归模型使用的数据集
    """
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    # 每隔 5 个样本就在样本的值上添加噪音
    y[::5] += 1 * (0.5 - np.random.rand(int(n / 5)))
    # 进行简单拆分，测试集大小占 1/4
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def data():
    """Fixture to create regression data."""
    return create_regression_data(1000)


def test_KNeighborsRegressor(data):
    """
    测试KNeighborsRegressor的用法
    :param data: 包含训练和测试样本及其对应标签的元组
    :return: None
    """
    X_train, X_test, y_train, y_test = data
    model = neighbors.KNeighborsRegressor()
    model.fit(X_train, y_train)
    print("\nTraining score: %f" % model.score(X_train, y_train))
    print("Testing score: %f" % model.score(X_test, y_test))


def test_KNeighborsRegressor_k_w(data):
    """
    测试KNeighborsRegressor的k和w参数
    :param data: 包含训练和测试样本及其对应标签的元组
    :return: None
    """
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 绘制不同weights下，预测得分随n_neighbors变化的曲线
    for weight in weights:
        training_scores = []
        testing_scores = []
        for K in Ks:
            model = neighbors.KNeighborsRegressor(n_neighbors=K, weights=weight)
            model.fit(X_train, y_train)
            training_scores.append(model.score(X_train, y_train))
            testing_scores.append(model.score(X_test, y_test))
        ax.plot(Ks, training_scores, label='Training score: %s' % weight)
        ax.plot(Ks, testing_scores, label='Testing score: %s' % weight)
    ax.legend(loc='best')
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('score')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('KNeighborsRegressor: k and w')
    plt.show()


def test_KNeighborsRegressor_k_p(data):
    """
    测试KNeighborsRegressor的k和p参数
    :param data: 包含训练和测试样本及其对应标签的元组
    :return: None
    """
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, endpoint=False, dtype='int')
    Ps = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 绘制不同weights下，预测得分随n_neighbors变化的曲线
    for P in Ps:
        training_scores = []
        testing_scores = []
        for K in Ks:
            model = neighbors.KNeighborsRegressor(n_neighbors=K, p=P)
            model.fit(X_train, y_train)
            training_scores.append(model.score(X_train, y_train))
            testing_scores.append(model.score(X_test, y_test))
        ax.plot(Ks, training_scores, label='Training score: p=%d' % P)
        ax.plot(Ks, testing_scores, label='Testing score: p=%d' % P, linestyle='--')
    ax.legend(loc='best')
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('score')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('KNeighborsRegressor: k and p')
    plt.show()


if __name__ == "__main__":
    # test_KNeighborsRegressor(data())
    # test_KNeighborsRegressor_k_w(data())
    test_KNeighborsRegressor_k_p(data())
