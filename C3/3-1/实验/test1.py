from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# 数据预处理
def load_and_preprocess_data():
    # # 加载手写数字数据集
    # digits = datasets.load_digits()
    #
    # # 数据特征：64维向量，每一行为一个样本
    # X = digits.data
    # # 标签：手写数字的真实分类
    # y = digits.target
    #
    # # 将数据集划分为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #
    # # 对特征进行归一化处理
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # return X_train, X_test, y_train, y_test, digits

    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, digits


# 模型训练
def train_knn_model(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


# 模型验证
def validate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)

    # 评估模型的准确性
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型的准确率: {accuracy * 100:.2f}%")

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:\n", conf_matrix)


# 主函数
if __name__ == "__main__":
    # 加载和预处理数据
    X_train, X_test, y_train, y_test, digits = load_and_preprocess_data()

    # 使用k=3训练模型
    knn_model = train_knn_model(X_train, y_train, k=3)

    # 验证模型
    validate_model(knn_model, X_test, y_test)

    # 可视化一个手写数字
    plt.figure(figsize=(3, 3))
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
