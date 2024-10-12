import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data  # 特征数据
y = diabetes.target  # 目标数据

# 将y调整成列向量
y = y.reshape(-1, 1)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 初始化权重 W 和偏置 b
W = np.zeros((X_train.shape[1], 1))  # 初始化为0，X_train.shape[1]是特征数
b = 0

# 初始预测值 h(x) 全为 0
y_pred = np.dot(X_train, W) + b  # 由于 W 和 b 都是 0，所以预测值为 0

# 计算初始损失值 (MSE)
initial_loss = (1 / (2 * len(y_train))) * np.sum((y_pred - y_train) ** 2)

print("糖尿病数据集的初始损失值:", initial_loss)
