from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


# 数据预处理
def load_and_preprocess_data():
    # 加载手写数字数据集
    digits = datasets.load_digits()

    # 数据特征：64维向量，每一行为一个样本
    X = digits.data
    # 标签：手写数字的真实分类
    y = digits.target

    # 将标签进行二值化，用于多分类的ROC
    y = label_binarize(y, classes=range(10))  # 0-9共10类
    n_classes = y.shape[1]

    # 对特征进行归一化处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, digits, n_classes


# 模型训练，使用最佳k值
def train_knn_model(X_train, y_train, k):
    # 使用 One-vs-Rest 将 KNeighborsClassifier 适应多分类问题
    knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=k))
    knn.fit(X_train, y_train)
    return knn


# 查找最佳的K值，使用交叉验证
def find_best_k(X, y):
    param_grid = {'estimator__n_neighbors': range(1, 10)}
    grid_search = GridSearchCV(OneVsRestClassifier(KNeighborsClassifier()), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print(f"最佳k值: {grid_search.best_params_['estimator__n_neighbors']}")
    return grid_search.best_params_['estimator__n_neighbors']


# 绘制多分类的ROC曲线
def plot_multiclass_roc(knn, X_test, y_test, n_classes):
    # 得到模型对每个类别的预测概率
    y_score = knn.predict_proba(X_test)

    # 计算每个类别的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 汇总宏观（macro）平均AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制ROC曲线
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    # 修正: 只保留虚线部分 '--'，并用 color 参数定义颜色
    plt.plot(fpr["macro"], tpr["macro"], '--',
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线，用于基准线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for multi-class')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # 加载和预处理数据
    X, y, digits, n_classes = load_and_preprocess_data()

    # 可视化一个手写数字
    # plt.figure(figsize=(3, 3))
    # plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 查找最佳k值
    best_k = find_best_k(X_train, y_train)

    # 使用最佳k值训练模型
    knn_model = train_knn_model(X_train, y_train, k=best_k)

    # 绘制多分类ROC和AUC曲线
    plot_multiclass_roc(knn_model, X_test, y_test, n_classes)
