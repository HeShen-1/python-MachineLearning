import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载加州房价数据集
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化和训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test)

# 计算并输出均方误差和R^2的分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.5f}")
print(f"R^2 Score: {r2:.5f}")

# 输出模型系数
'''
    线性回归模型的系数表示每个特征对房价中位数的影响程度。系数的绝对值越大，影响程度越大。
'''
coefficients = pd.DataFrame(model.coef_, california_housing.feature_names, columns=['Coefficient'])
print(coefficients)

# 可视化实际值与预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)      # 绘制散点图，alpha参数控制透明度，0为完全透明，1为完全不透明
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')   # 对角线
plt.show()
