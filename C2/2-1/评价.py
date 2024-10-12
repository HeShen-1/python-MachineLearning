from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 导入数据集
iris = load_iris()

# x存放特征数据，y存放标签数据
x = iris.data
y = iris.target

# 划分样本集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=1, shuffle=True, stratify=y)
# 构建模型
model = neighbors.KNeighborsClassifier(n_neighbors=3)

# 拟合训练集
model.fit(x_train, y_train)

# 对测试集进行预测
prediction = model.predict(x_test)

# print(prediction)
# print(y_test)

print(classification_report(y_true=y_test, y_pred=prediction))
