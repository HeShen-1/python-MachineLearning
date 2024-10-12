# 导入所需库
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 导入分类模型
from sklearn.linear_model import LogisticRegression     # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier      # K近邻
from sklearn.svm import SVC                             # 支持向量机
from sklearn.ensemble import RandomForestClassifier     # 随机森林
from sklearn.tree import DecisionTreeClassifier         # 决策树

# 导入数据集
data = load_breast_cancer()
X = data.data  # 特征
y = data.target  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义分类器
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# 训练并评估模型
for model_name, model in models.items():
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测测试集
    accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
    print(f'{model_name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    print('-' * 50)
