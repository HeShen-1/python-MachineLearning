import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


# Iris 数据集中的目标变量 y 是整数向量（类标签），
# 但逻辑回归模型期望 y 为二进制格式（0 和 1）以进行二进制分类。需要相应地转换标签。
class LogisticRegressionBinary(object):
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


# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 鸢尾花数据集有三个类（0，1，2），筛选出类 0 和 1 的数据
# 对于二元分类，筛选类 0 和 1 的数据
binary_filter = y < 2
X = X[binary_filter]
y = y[binary_filter].reshape((-1, 1))

# 在 X 矩阵左侧添加全 1 的列，说明模型中的截距项
one = np.ones((X.shape[0], 1))
X_train = np.hstack((one, X))

# 训练 Logistic 回归模型，使用 Logistic 回归进行二进制分类。
classify = LogisticRegressionBinary()
loss = classify.train(X_train, y)

# 输出学习的权重
print("Learned weights:\n", classify.W)

# 绘制迭代的损失曲线
plt.plot(loss)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.title('Loss curve for Logistic Regression')
plt.show()
