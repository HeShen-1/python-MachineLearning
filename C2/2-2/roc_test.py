import numpy as np
import matplotlib.pyplot as plt
from Classification_func import *
from sklearn.metrics import roc_curve, auc

y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
y_hat = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1])

print("TPR:", TPR(y, y_hat))
print("FPR:", FPR(y, y_hat))
print("FNR:", FNR(y, y_hat))
print("TNR:", TNR(y, y_hat))
print("Accuracy:", ACC(y, y_hat))
print("Precision:", PRE(y, y_hat))
print("Recall:", REC(y, y_hat))
print("F1_Score:", F1_score(y, y_hat))

print("-----------------------------------")
# points = ROC(y, y_hat)
# df = pd.DataFrame(points, columns=['tpr', 'fpr'])
print("AUC is %.3f." % AUC(y, y_hat))
# df.plot(x='fpr', y='tpr', label="ROC", xlabel="FPR", ylabel="TPR")
# # print(df)
# plt.show()


# Calculate ROC curve and AUC
fpr_values, tpr_values, _ = roc_curve(y, y_hat)
roc_auc = auc(fpr_values, tpr_values)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
