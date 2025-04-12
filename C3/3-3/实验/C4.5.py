import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus


def outlook_type(s):
    """
    将天气类型转换为数字
    :param s: 天气类型
    :return: 数字表示的天气类型
    """
    it = {b'sunny': 0, b'overcast': 1, b'rainy': 2}
    return it[s]


def temp_type(s):
    """
    将温度类型转换为数字
    :param s: 温度类型
    :return: 数字表示的温度类型
    """
    it = {b'hot': 0, b'mild': 1, b'cool': 2}
    return it[s]


def humidity_type(s):
    """
    将湿度类型转换为数字
    :param s: 湿度类型
    :return: 数字表示的湿度类型
    """
    it = {b'high': 0, b'normal': 1}
    return it[s]


def windy_type(s):
    """
    将风速类型转换为数字
    :param s: 风速类型
    :return: 数字表示的风速类型
    """
    it = {b'FALSE': 0, b'TRUE': 1}
    return it[s]


def play_type(s):
    """
    将是否打球类型转换为数字
    :param s: 是否打球类型
    :return: 数字表示的是否打球类型
    """
    it = {b'no': 0, b'yes': 1}
    return it[s]


play_feature_E = 'outlook', 'temperature', 'humidity', 'windy'
play_class = 'yes', 'no'

# 1.读取数据集，并将原始数据中的数据转化为数字
data = np.loadtxt('C3/3-3/实验/play.txt', dtype=str, delimiter='\t', skiprows=1, converters={0: outlook_type, 1: temp_type, 2: humidity_type, 3: windy_type, 4: play_type})
x, y = np.split(data, (4, ), axis=1)  # 将数据集分为特征和标签

# 2.划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # 70%训练集，30%测试集

# 3.使用信息熵作为划分标准，构建决策树
clf = tree.DecisionTreeClassifier(criterion='entropy')  # 创建决策树分类器
print("clf:", clf)  # 打印决策树分类器
clf = clf.fit(x_train, y_train)  # 拟合训练数据

# 4.把决策树结构写入文件
dot_data = tree.export_graphviz(
    clf, out_file=None, 
    feature_names=play_feature_E, class_names=play_class, filled=True, rounded=True, special_characters=True)  # 导出决策树结构
graph = pydotplus.graph_from_dot_data(dot_data)  # 创建图形对象
graph.write_pdf('C3/3-3/实验/play.pdf')  # 将决策树结构写入PDF文件

# 系数反应每个特征的影响力，越大越重要
print("特征重要性系数:", clf.feature_importances_)  # 打印特征重要性系数
print("测试集准确率:",  clf.score(x_test, y_test))  # 打印测试集准确率

print("测试集预测结果:", clf.predict(x_test))  # 打印测试集预测结果

# 5.使用训练数据进行预测
answer = clf.predict(x_train)
y_train = y_train.reshape(-1)
print("训练集预测结果:", answer)  # 打印训练集预测结果
print("训练集真实结果:", y_train)  # 打印训练集真实结果
print("训练集准确率:", np.mean(answer == y_train))  # 打印训练集准确率
print("训练集准确率", clf.score(x_train, y_train))  # 打印训练集准确率

# 6.对测试数据进行预测
answer = clf.predict(x_test)  # 对测试数据进行预测
y_test = y_test.reshape(-1)  # 将测试数据标签展平
print("测试集预测结果:", answer)  # 打印测试集预测结果
print("测试集真实结果:", y_test)  # 打印测试集真实结果
print("测试集准确率:", np.mean(answer == y_test))  # 打印测试集准确率