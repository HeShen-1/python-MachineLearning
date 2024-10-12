# 导入鸢尾花数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 导入数据集
iris = load_iris()

# x存放特征数据，y存放标签数据
x = iris.data
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=1, shuffle=True, stratify=y)
# 计算分层采样时各类的抽样占比
count = Counter(y_train)
print("使用分层采样")
for k, v in count.items():
    ratio = v / len(y_train)
    print("class:{0}, radio:{1:.2f}".format(k, ratio))
