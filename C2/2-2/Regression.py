import numpy as np

from sklearn.metrics import mean_squared_error      # MSE:均方误差
from sklearn.metrics import mean_absolute_error     # MAE:平均绝对误差
from sklearn.metrics import r2_score                # R2得分

# 数据集
y_test = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_predict = np.array([1.0, 4.5, 3.8, 3.2, 3.0, 4.8, -2.2])

mean_squared_error(y_test, y_predict)
mean_absolute_error(y_test, y_predict)
np.sqrt(mean_squared_error(y_test, y_predict))      # RMSE:均方根误差
r2_score(y_test, y_predict)

print("MAE:", mean_absolute_error(y_test, y_predict))
print("MSE:", mean_squared_error(y_test, y_predict))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_predict)))
print("R2:", r2_score(y_test, y_predict))

