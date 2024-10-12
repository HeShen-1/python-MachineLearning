import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


class logistic(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, lr=0.01, num_iters=5000):
        num_train, num_feature = X.shape
        self.W = 0.001 * np.random.randn(num_feature, 1).reshape((-1, 1))
        loss = []
        for i in range(num_iters):
            error, dW = self.compute_loss(X, y)
            self.W += - lr * dW
            loss.append(error)
            if i % 200 == 0:
                print('i= %d, error= %f' % (i, error))
        return loss

    def compute_loss(self, X, y):
        num_train = X.shape[0]
        h = self.output(X)
        loss = - np.sum((y * np.log(h) + (1 - y) * np.log(1 - h))) / num_train
        dW = X.T.dot(h - y) / num_train
        return loss, dW

    def output(self, X):
        g = np.dot(X, self.W)
        return self.sigmoid(g)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, X_test):
        h = self.output(X_test)
        return np.where(h >= 0.5, 1, 0)


iris = load_iris()
data = iris.data
target = iris.target
X = data[0:100, [0, 2]]
y = target[0:100]
y = y.reshape((-1, 1))
one = np.ones((X.shape[0], 1))
X_train = np.hstack((one, X))
classify = logistic()
loss = classify.train(X_train, y)
label = np.array(y)
index_0 = np.where(label == 0)
plt.scatter(X[index_0, 0], X[index_0, 1], marker='x', c='b', label='0', s=15)
index_1 = np.where(label == 1)
plt.scatter(X[index_1, 0], X[index_1, 1], marker='o', c='r', label='1', s=15)
# 绘制分类边界线
x1 = np.arange(4, 7.5, 0.5)
x2 = (- classify.W[0] - classify.W[1] * x1) / classify.W[2]
plt.plot(x1, x2, color='black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper left')
plt.show()
