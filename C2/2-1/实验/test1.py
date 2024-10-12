from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.utils import resample

# 加载乳腺癌数据集
cancer = load_breast_cancer()
# print(cancer.DESCR)

# 提取特征和标签
x = cancer.data
y = cancer.target

# 生成自助法样本
n_samples = len(x)
x_train, y_train = resample(x, y, n_samples=n_samples, replace=True)  # 生成训练集（有放回抽样）
x_test = np.array([x for x in x if x.tolist() not in x_train.tolist()])  # 生成测试集（不在训练集中的样本）
y_test = np.array([y[i] for i, x in enumerate(x) if x.tolist() not in x_train.tolist()])

# 构建模型
model = neighbors.KNeighborsClassifier(n_neighbors=3)

# 拟合训练集
model.fit(x_train, y_train)

# 对测试集进行预测
prediction = model.predict(x_test)

print("混淆矩阵：\n", confusion_matrix(y_true=y_test, y_pred=prediction))
print("分类报告：\n", classification_report(y_true=y_test, y_pred=prediction))
