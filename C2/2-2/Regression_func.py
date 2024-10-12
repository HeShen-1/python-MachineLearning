import numpy as np


def MAE(y_true, y_pred):
    n = len(y_true)
    return np.sum(np.abs(y_true - y_pred)) / n


def MAPE(y_true, y_pred):
    n = len(y_true)
    return np.sum(np.abs((y_true - y_pred) / y_true) / n)


def MSE(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))


def R2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


# 数据集
y_test = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_predict = np.array([1.0, 4.5, 3.8, 3.2, 3.0, 4.8, -2.2])

print("MAE:", MAE(y_test, y_predict))
print("MAPE:", MAPE(y_test, y_predict))
print("MSE:", MSE(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))
print("R2:", R2(y_test, y_predict))
