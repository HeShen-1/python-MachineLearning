from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np


# 数据预处理
def load_and_preprocess_data():
    # 加载手写数字数据集
    digits = datasets.load_digits()

    # 数据特征：64维向量，每一行为一个样本
    X = digits.data
    # 标签：手写数字的真实分类
    y = digits.target

    # 对特征进行归一化处理
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, digits


# 模型训练，使用交叉验证
def cross_validate_knn(X, y, k):
    knn = KNeighborsClassifier(n_neighbors=k)

    # 设置StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 进行5折交叉验证
    cv_scores = cross_val_score(knn, X, y, cv=skf, scoring='accuracy')

    # 输出交叉验证的平均准确率
    for i, score in enumerate(cv_scores):
        print(f"第{i+1}折交叉验证的准确率: {score * 100:.3f}%")
    print(f"5折交叉验证的平均准确率: {np.mean(cv_scores) * 100:.3f}%")

    # 返回训练好的模型
    knn.fit(X, y)
    return knn


# 模型验证
def validate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]  # 用于计算ROC和PRC曲线

    # 评估模型的准确性
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型的准确率: {accuracy * 100:.2f}%")

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("混淆矩阵:\n", conf_matrix)
    print("分类报告:\n", classification_report(y_test, y_pred))

    # 计算准确率、精确率、召回率、F1值
    """
    average 参数有以下几种选择：
        'micro': 计算全局的指标，忽略类别。
        'macro': 对每个类别计算指标后取平均，不考虑类别的不平衡。
        'weighted': 对每个类别计算指标后取加权平均，考虑类别的不平衡。
        None: 返回每个类别的结果，而不是整体的指标。
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # # 计算ROC AUC
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # roc_auc = auc(fpr, tpr)
    # print(f"ROC AUC: {roc_auc:.2f}")
    #
    # # 计算PRC AUC
    # precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    # prc_auc = auc(recall_vals, precision_vals)
    # print(f"PRC AUC: {prc_auc:.2f}")


# 主函数
if __name__ == "__main__":
    # 加载和预处理数据
    X, y, digits = load_and_preprocess_data()

    # 使用交叉验证和最佳k值训练模型
    knn_model = cross_validate_knn(X, y, k=3)

    # 划分测试集以验证模型效果
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    validate_model(knn_model, X_test, y_test)

    # 可视化一个手写数字
    # plt.figure(figsize=(3, 3))
    # plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()
