from sklearn import neighbors, datasets, model_selection
import pytest
import numpy as np
import matplotlib.pyplot as plt


def load_classification_data():
    """
    加载分类模型使用的数据集
    :return: 一个元组，依次为训练样本集、测试样本集、训练样本的标记、测试样本的标记
    """
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    return model_selection.train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)


@pytest.fixture
def data():
    """Fixture to load classification data."""
    return load_classification_data()


def test_KNeighborsClassifier(data):
    """
    测试KNeighborsClassifier分类器的用法
    :param data: 包含训练和测试样本及其对应标签的元组
    """
    X_train, X_test, y_train, y_test = data
    model = neighbors.KNeighborsClassifier()
    model.fit(X_train, y_train)
    print("\nTraining score: %f" % model.score(X_train, y_train))
    print("Testing score: %f" % model.score(X_test, y_test))


def test_KNeighborsClassifier_k_w(data):
    """
    测试KNeighborsClassifier分类器的n_neighbors和weights参数的影响
    :param data: 包含训练和测试样本及其对应标签的元组
    :return: None
    """
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 绘制不同weights下，不同K值对应的训练和测试准确率
    for weight in weights:
        training_scores = []
        testing_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(n_neighbors=K, weights=weight)
            clf.fit(X_train, y_train)
            training_scores.append(clf.score(X_train, y_train))
            testing_scores.append(clf.score(X_test, y_test))
        ax.plot(Ks, training_scores, label='Training score: %s' % weight)
        ax.plot(Ks, testing_scores, label='Testing score: %s' % weight)
    ax.legend(loc='best')
    ax.set_title('KNeighborsClassifier: k and weights')
    ax.set_xlabel('k')
    ax.set_ylabel('score')
    ax.set_ylim([0.0, 1.05])
    plt.show()


def test_KNeighborsClassifier_k_p(data):
    """
    测试KNeighborsClassifier分类器的p参数的影响
    :param data: 包含训练和测试样本及其对应标签的元组
    :return: None
    """
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, endpoint=False, dtype='int')
    Ps = [1, 2, 10]

    flg = plt.figure()
    ax = flg.add_subplot(1, 1, 1)
    # 绘制不同p值下，不同K值对应的训练和测试准确率
    for P in Ps:
        training_scores = []
        testing_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(n_neighbors=K, p=P)
            clf.fit(X_train, y_train)
            training_scores.append(clf.score(X_train, y_train))
            testing_scores.append(clf.score(X_test, y_test))
        ax.plot(Ks, training_scores, label='Training score:P=%d' % P)
        ax.plot(Ks, testing_scores, label='Testing score: P=%d' % P)
    ax.legend(loc='best')
    ax.set_title('KNeighborsClassifier: k and p')
    ax.set_xlabel('k')
    ax.set_ylabel('score')
    ax.set_ylim([0.0, 1.05])
    plt.show()


if __name__ == "__main__":
    # test_KNeighborsClassifier(data())
    # test_KNeighborsClassifier_k_w(data())
    test_KNeighborsClassifier_k_p(data())
