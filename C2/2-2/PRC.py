import warnings

import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()
x = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=666)

# 忽略警告信息
warnings.filterwarnings('ignore')

# 创建逻辑回归模型，设置参数
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, random_state=666)
log_reg.fit(x_train, y_train)
decision_scores = log_reg.decision_function(x_test)

precision, recall, thresholds = precision_recall_curve(y_test, decision_scores)
# print("Precision: ", precision)
# print("Recall: ", recall)
# print("Thresholds: ", thresholds)

# 预测
y_pred = log_reg.predict(x_test)

# 模型评估
accuracy = np.mean(y_pred == y_test)
print("Accuracy: ", accuracy)

# 绘制Precision-Recall曲线
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.plot(recall, precision)
plt.show()
