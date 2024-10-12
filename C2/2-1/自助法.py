import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

# 创建DataFrame，将数据放入
data = {
    "SLength": x[:, 0],
    "SWidth": x[:, 1],
    "PLength": x[:, 2],
    "PWidth": x[:, 3],
}

data = pd.DataFrame(data)

# 将标签作为一列添加进去
data['class'] = y

# 随机采样得到训练集
train_data = data.sample(frac=1.0, replace=True)

# 按照索引定位，找到data中不在训练集中的数据作为测试集
test_data = data.loc[data.index.difference(train_data.index)].copy()

print("训练集")
print("index", train_data)
print("测试集")
print("index", test_data)
