import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import seed

from Classification_func import *

# 对苹果的态度
# y = [1, 1, 0, 0, 1]

# 将苹果的图片等特征带入分类模型中得到的分类结果
# y_hat = [1, 1, 0, 0, 0]

z = np.array([1, 1, 0, 0, 1])
z_hat = np.array([1, 1, 0, 0, 0])

print("TPR:", TPR(z, z_hat))
print("FPR:", FPR(z, z_hat))
print("FNR:", FNR(z, z_hat))
print("TNR:", TNR(z, z_hat))
print("Accuracy:", ACC(z, z_hat))
print("Precision:", PRE(z, z_hat))
print("Recall:", REC(z, z_hat))
print("F1_Score:", F1_score(z, z_hat))

# 绘制ROC曲线
seed(15)
y = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0])
y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35,
                   0.34, 0.33, 0.3, 0.1])
points = ROC(y, y_pred)
df = pd.DataFrame(points, columns=['tpr', 'fpr'])
print("AUC is %.3f." % AUC(y, y_pred))
df.plot(x='fpr', y='tpr', label="ROC", xlabel="FPR", ylabel="TPR")
# print(df)
plt.show()
