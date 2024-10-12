import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../iris_dataset.csv', header=0)  # 修改为 header=0
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

X = data.iloc[0:150, [0, 2]].values  # 获取 sepal length 和 petal length
y = data.iloc[0:150, 4].values
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2

X_setosa, y_setosa = X[0:50], y[0:50]
X_versicolor, y_versicolor = X[50:100], y[50:100]
X_virginica, y_virginica = X[100:150], y[100:150]

plt.scatter(X_setosa[:, 0], X_setosa[:, 1], color='red', marker='o', label='setosa')
plt.scatter(X_versicolor[:, 0], X_versicolor[:, 1], color='blue', marker='^', label='versicolor')
plt.scatter(X_virginica[:, 0], X_virginica[:, 1], color='green', marker='s', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

