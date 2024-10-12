# 真正率
def TPR(y, y_hat):
    true_positive = sum(yi == 1 and yi_hat == 1 for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive


# 假正率
def FPR(y, y_hat):
    false_positive = sum(yi == 0 and yi_hat == 1 for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return false_positive / actual_negative


# 假负率
def FNR(y, y_hat):
    false_negative = sum(yi == 1 and yi_hat == 0 for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return false_negative / actual_positive


# 真负率
def TNR(y, y_hat):
    false_negative = sum(yi == 0 and yi_hat == 0 for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return false_negative / actual_negative


# 准确率
def ACC(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)


# 精确率
def PRE(y, y_hat):
    true_positive = sum(yi == 1 and yi_hat == 1 for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_positive / predicted_positive


# 召回率
def REC(y, y_hat):
    true_positive = sum(yi == 1 and yi_hat == 1 for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive


# F1值
def F1_score(y, y_hat):
    return 2 * (PRE(y, y_hat) * REC(y, y_hat)) / (PRE(y, y_hat) + REC(y, y_hat))


# ROC曲线
def ROC(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([TPR(y, y_hat), 1 - TNR(y, y_hat)])
    return ret


# AUC曲线
def AUC(y, y_hat_prob):
    roc = iter(ROC(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        # tpr_pre, fpr_pre = tpr, fpr
        tpr_pre = tpr
        fpr_pre = fpr
    return auc
