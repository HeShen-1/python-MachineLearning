import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 只保留 Setosa 和 Versicolor 两类 (target == 0 是 Setosa, target == 1 是 Versicolor)
mask = y < 2
X = X[mask]
y = y[mask]

# 目标变量转换为 0 和 1
y = (y == 0).astype(int)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 定义 Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义梯度计算函数
def compute_gradient(X, y, theta):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    gradient = (X.T @ (h - y)) / m
    return gradient


# 梯度下降法实现
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
    return theta


# 定义 Hessian 计算函数
def compute_hessian(X, theta):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    R = np.diag(h * (1 - h))
    H = X.T @ R @ X / m
    return H


# 牛顿法实现
def newton_method(X, y, theta, num_iterations):
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta)
        hessian = compute_hessian(X, theta)
        theta = theta - np.linalg.inv(hessian) @ gradient
    return theta


# 预测函数
def predict(X, theta):
    return (sigmoid(X @ theta) >= 0.5).astype(int)


# 交叉验证
def cross_validate_model(X, y, method, num_folds=5, learning_rate=0.01, num_iterations=1000):
    kf = KFold(n_splits=num_folds)
    accuracies = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        theta = np.zeros(X_train.shape[1])

        # 根据选择的方法优化参数
        if method == 'gd':
            theta_optimal = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)
        elif method == 'newton':
            theta_optimal = newton_method(X_train, y_train, theta, num_iterations=10)

        # 在验证集上进行预测
        y_pred = predict(X_val, theta_optimal)

        # 计算准确率
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


# 绘制分类边界
def plot_decision_boundary(X, y, theta, title):
    # 创建网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # 计算网格点的预测值
    Z = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ theta)
    Z = Z.reshape(xx.shape)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制决策边界
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.xlabel('花萼长度')
    plt.ylabel('花萼宽度')
    plt.title(title)
    plt.show()


# 主程序
if __name__ == "__main__":
    # 使用交叉验证对梯度下降和牛顿法进行验证
    accuracy_gd = cross_validate_model(X, y, method='gd', num_folds=5)
    accuracy_newton = cross_validate_model(X, y, method='newton', num_folds=5)

    print(f'梯度下降法在交叉验证中的平均准确率: {accuracy_gd:.9f}')
    print(f'牛顿迭代法在交叉验证中的平均准确率: {accuracy_newton:.9f}')

    # 选择前两个特征进行可视化
    X_selected = X[:, :2]  # 取前两个特征
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # 使用梯度下降法优化
    theta = np.zeros(X_train.shape[1])
    theta_optimal_gd = gradient_descent(X_train, y_train, theta, learning_rate=0.01, num_iterations=1000)

    # 使用牛顿法优化
    theta_optimal_newton = newton_method(X_train, y_train, theta, num_iterations=10)

    # 绘制分类边界
    plot_decision_boundary(X_train, y_train, theta_optimal_gd, '梯度下降法分类边界')
    plot_decision_boundary(X_train, y_train, theta_optimal_newton, '牛顿迭代法分类边界')
