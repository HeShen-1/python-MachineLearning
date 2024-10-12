import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize

# 加载手写数字数据集
digits = load_digits()
X, y = digits.data, digits.target

# 将标签二值化，适用于多分类问题
y_bin = label_binarize(y, classes=np.arange(10))

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# 数据归一化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练K近邻模型，使用One-vs-Rest方法
knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
knn.fit(X_train_scaled, y_train)

# 预测
y_score = knn.predict_proba(X_test_scaled)

# 计算 AUC for each class
roc_auc_dict = {}
for i in range(10):
    roc_auc = roc_auc_score(y_test[:, i], y_score[:, i])
    roc_auc_dict[f"Class {i}"] = roc_auc
    print(f"AUC for Class {i}: {roc_auc:.3f}")

# 计算平均AUC
mean_auc = np.mean(list(roc_auc_dict.values()))
print(f"Mean AUC: {mean_auc:.3f}")

# 分类报告（基于硬预测）
y_pred = knn.predict(X_test_scaled)
y_test_orig = np.argmax(y_test, axis=1)  # 转回原始标签
y_pred_orig = np.argmax(y_pred, axis=1)

# 输出分类报告
report = classification_report(y_test_orig, y_pred_orig, target_names=[str(i) for i in range(10)])
print("\nClassification Report:\n")
print(report)

# 初始化绘图
fig, (ax_prc, ax_roc) = plt.subplots(1, 2, figsize=(14, 7))

# 为每个类别绘制 ROC 和 PRC 曲线
for i in range(10):
    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f'Class {i}').plot(ax=ax_roc)

    # 计算 PRC 曲线
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    avg_precision = average_precision_score(y_test[:, i], y_score[:, i])
    PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision,
                           estimator_name=f'Class {i}').plot(ax=ax_prc)

# 设置图形标题
ax_roc.set_title('ROC Curves for Each Class')
ax_prc.set_title('PRC Curves for Each Class')

# 显示图形
plt.tight_layout()
plt.show()
