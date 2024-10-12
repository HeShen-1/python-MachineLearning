from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from collections import Counter

# 导入数据集
iris = load_iris()

# x存放特征数据，y存放标签数据
x = iris.data
y = iris.target

# 十折交叉验证
sp = 10
x_train, x_test, y_train, y_test = [], [], [], []

# 划分训练集和测试集
skf = StratifiedKFold(n_splits=sp, shuffle=True, random_state=1)

# 遍历索引生成器获取每次划分的训练集和测试集‘
for train_index, test_index in skf.split(x, y):
    x_test = x[test_index]
    y_test = y[test_index]
    x_train = x[train_index]
    y_train = y[train_index]

# 计算未使用分层采样时各类的抽样占比
i = 0
count = Counter(y_train)
for train_index, test_index in skf.split(x, y):
    if i == 3:
        break
    print("使用分层采样")
    for k, v in count.items():
        ratio = v / len(y_train)
        print("class:{0}, radio:{1:.2f}".format(k, ratio))
    i += 1
