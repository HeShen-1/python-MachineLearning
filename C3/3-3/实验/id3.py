from math import log
import operator
import treePlotter

def createDataSet():
    """
    创建数据集
    :return: 数据集和标签
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算信息熵
    :param dataSet: 数据集
    :return: 信息熵
    """
    numEntries = len(dataSet)
    labelCounts = {}            # 创建字典，存储每个标签的出现次数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 特征索引
    :param value: 特征值
    :return: 划分后的数据集
    """
    retDataSet = []            # 创建返回的数据集列表
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]         # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 连接剩余特征
            retDataSet.append(reducedFeatVec)       # 添加到返回的数据集
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet: 数据集
    :return: 最佳划分特征的索引
    """
    numFeatures = len(dataSet[0]) - 1          # 特征数
    baseEntropy = calcShannonEnt(dataSet)      # 计算当前数据集的信息熵
    bestInfoGain = 0.0                         # 初始化最佳信息增益
    bestFeature = -1                           # 初始化最佳特征索引
    for i in range(numFeatures):                # 遍历所有特征
        featList = [example[i] for example in dataSet]  # 创建特征列表
        uniqueVals = set(featList)              # 创建唯一特征值集合
        newEntropy = 0.0                        # 初始化新的信息熵
        for value in uniqueVals:                # 遍历所有特征值
            subDataSet = splitDataSet(dataSet, i, value)  # 划分数据集
            prob = len(subDataSet) / float(len(dataSet))   # 计算概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 更新新的信息熵
        infoGain = baseEntropy - newEntropy     # 计算信息增益
        if infoGain > bestInfoGain:             # 如果信息增益更大，更新最佳信息增益和最佳特征索引
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                          # 返回最佳特征索引


def majorityCnt(classList):
    """
    统计出现次数最多的类标签
    :param classList: 类标签列表
    :return: 出现次数最多的类标签
    """
    classCount = {}                            # 创建字典，存储每个类标签的出现次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按照出现次数排序
    return sortedClassCount[0][0]              # 返回出现次数最多的类标签


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 特征标签
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet]  # 获取类标签列表
    if classList.count(classList[0]) == len(classList):  # 如果所有类标签都相同，返回该类标签
        return classList[0]
    if len(dataSet[0]) == 1:                             # 如果没有特征可划分，返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)         # 选择最佳特征索引
    bestFeatLabel = labels[bestFeat]                     # 获取最佳特征标签
    myTree = {bestFeatLabel: {}}                         # 创建决策树字典
    del(labels[bestFeat])                                # 删除已使用的特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 获取最佳特征值列表
    uniqueVals = set(featValues)                         # 创建唯一特征值集合
    for value in uniqueVals:                             # 遍历所有特征值
        subLabels = labels[:]                            # 创建副本，避免修改原始列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 递归创建子树
    return myTree                                        # 返回决策树


def classify(inputTree, featLabels, testVec):
    """
    使用决策树进行分类
    :param inputTree: 决策树
    :param featLabels: 特征标签
    :param testVec: 测试向量
    :return: 分类结果
    """
    firstStr = list(inputTree.keys())[0]                # 获取根节点特征标签
    secondDict = inputTree[firstStr]                    # 获取根节点对应的子树
    featIndex = featLabels.index(firstStr)              # 获取根节点特征索引
    for key in secondDict.keys():                       # 遍历子树的所有特征值
        if testVec[featIndex] == key:                   # 如果测试向量的特征值与子树的特征值相同
            if isinstance(secondDict[key], dict):       # 如果子树是字典，递归调用 classify 函数
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:                                       # 如果子树是叶子节点，返回类标签
                classLabel = secondDict[key]
    return classLabel                                    # 返回分类结果


def storeTree(inputTree, filename):
    """
    存储决策树
    :param inputTree: 决策树
    :param filename: 文件名
    """
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)                      # 使用 pickle 序列化决策树并存储到文件
    fw.close()                                          # 关闭文件


def grabTree(filename):
    """
    从文件中读取决策树
    :param filename: 文件名
    :return: 决策树
    """
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)                          # 使用 pickle 反序列化决策树并返回
    fr.close()                                          # 关闭文件


if __name__ == '__main__':
    fr = open('C3\\3-3\\实验\\play.txt')  # 打开数据集文件
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 读取数据集
    fr.close()                                          # 关闭文件
    print(lenses)
    lensesLabels = ['outlook', 'temperature', 'huminidy', 'windy']  # 特征标签
    lensesTree = createTree(lenses, lensesLabels)      # 创建决策树
    print(lensesTree)                                  # 打印决策树
    treePlotter.createPlot(lensesTree)                # 绘制决策树