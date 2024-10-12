import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../iris_dataset.csv', header=0)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

X = data.iloc[0:150, [0, 2]].values
y = data.iloc[0:150, 4].values
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2

X_setosa, y_setosa = X[0:50], y[0:50]
X_versicolor, y_versicolor = X[50:100], y[50:100]
X_virginica, y_virginica = X[100:150], y[100:150]

X_train = np.vstack((X_setosa[0:30, :], X_versicolor[0:30, :], X_virginica[0:30, :]))
y_train = np.hstack((y_setosa[0:30], y_versicolor[0:30], y_virginica[0:30]))

X_val = np.vstack((X_setosa[30:40, :], X_versicolor[30:40, :], X_virginica[30:40, :]))
y_val = np.hstack((y_setosa[30:40], y_versicolor[30:40], y_virginica[30:40]))

X_test = np.vstack((X_setosa[40:50, :], X_versicolor[40:50, :], X_virginica[40:50, :]))
y_test = np.hstack((y_setosa[40:50], y_versicolor[40:50], y_virginica[40:50]))


class KNearestNeighbor(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # 计算欧氏距离
        dists = np.zeros((num_test, num_train))

        dists = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(self.X_train ** 2, axis=1) - 2 * np.dot(X,
                                                                                                       self.X_train.T)

        # 将负值置为0，以避免 sqrt 计算无效值
        dists[dists < 0] = 0

        dists = np.sqrt(dists)

        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dist_k_min = np.argsort(dists[i])[:k]
            y_kclose = self.y_train[dist_k_min]
            y_pred[i] = np.argmax(np.bincount(y_kclose))

        return y_pred


# KNN = KNearestNeighbor()
# KNN.train(X_train, y_train)
# y_pred = KNN.predict(X_test, k=3)
# accuracy = np.mean(y_pred == y_test)
# print('Accuracy: {:.2f}'.format(accuracy))

# 训练集
X_setosa_train = X_setosa[0:30, :]
X_versicolor_train = X_versicolor[0:30, :]
X_virginica_train = X_virginica[0:30, :]
plt.scatter(X_setosa_train[:, 0], X_setosa_train[:, 1], color='red', marker='o', label='setosa_train')
plt.scatter(X_versicolor_train[:, 0], X_versicolor_train[:, 1], color='green', marker='^', label='versicolor_train')
plt.scatter(X_virginica_train[:, 0], X_virginica_train[:, 1], color='blue', marker='s', label='virginica_train')

# 测试集
X_setosa_test = X_setosa[40:50, :]
X_versicolor_test = X_versicolor[40:50, :]
X_virginica_test = X_virginica[40:50, :]
plt.scatter(X_setosa_test[:, 0], X_setosa_test[:, 1], color='y', marker='o', label='setosa_test')
plt.scatter(X_versicolor_test[:, 0], X_versicolor_test[:, 1], color='y', marker='^', label='versicolor_test')
plt.scatter(X_virginica_test[:, 0], X_virginica_test[:, 1], color='y', marker='s', label='virginica_test')
plt.legend(loc=4)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.show()
