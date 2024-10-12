import matplotlib.pyplot as plt

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# 拼合图像
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 创建分类器：支持向量分类器
clf = svm.SVC(gamma=0.001)

# 将数据拆分为 50% 的训练子集和 50% 的测试子集
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# 了解 train 子集上的数字
clf.fit(X_train, y_train)

# 预测测试子集上的数字值
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

# # 基本实况和预测列表
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix
#
# # 对于混淆矩阵中的每个单元格，添加相应的真实值和预测
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]
#
# print("Classification report rebuilt from confusion matrix:\n"f"{metrics.classification_report(y_true, y_pred)}\n")
