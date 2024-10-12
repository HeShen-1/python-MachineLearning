import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

actual = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
predicted = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1])

# 更改拉取请求

# 计算混淆矩阵分量
tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()

# 计算指标
tpr = tp / (tp + fn)  # True Positive Rate (Recall)
fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate
tnr = tn / (tn + fp)  # True Negative Rate

accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)
f1 = f1_score(actual, predicted)

# 计算 ROC 曲线和 AUC
fpr_values, tpr_values, _ = roc_curve(actual, predicted)
roc_auc = auc(fpr_values, tpr_values)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
