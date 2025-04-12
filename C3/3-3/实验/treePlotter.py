import matplotlib.pyplot as plt

decisonNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    """
    计算叶子节点的数量
    :param myTree: 决策树
    :return: 叶子节点的数量
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]  # 获取根节点特征标签
    secondDict = myTree[firstStr]      # 获取根节点对应的子树
    for key in secondDict.keys():      # 遍历子树的所有特征值
        if isinstance(secondDict[key], dict):  # 如果子树是字典，递归调用 getNumLeafs 函数
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1                # 如果子树是叶子节点，计数加一
    return numLeafs                    # 返回叶子节点的数量


def getTreeDepth(myTree):
    """
    计算决策树的深度
    :param myTree: 决策树
    :return: 决策树的深度
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]  # 获取根节点特征标签
    secondDict = myTree[firstStr]      # 获取根节点对应的子树
    for key in secondDict.keys():      # 遍历子树的所有特征值
        if isinstance(secondDict[key], dict):  # 如果子树是字典，递归调用 getTreeDepth 函数
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1                # 如果子树是叶子节点，深度为1
        if thisDepth > maxDepth:        # 更新最大深度
            maxDepth = thisDepth
    return maxDepth                    # 返回决策树的深度


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制节点
    :param nodeTxt: 节点文本
    :param centerPt: 节点中心坐标
    :param parentPt: 父节点坐标
    :param nodeType: 节点类型
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)  # 绘制节点文本
    

def plotMidText(cntrPt, parentPt, txtString):
    """
    绘制节点文本
    :param cntrPt: 节点中心坐标
    :param parentPt: 父节点坐标
    :param txtString: 文本字符串
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算文本的x坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 计算文本的y坐标
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)         # 绘制文本


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree: 决策树
    :param parentPt: 父节点坐标
    :param nodeTxt: 节点文本
    """
    numLeafs = getNumLeafs(myTree)                      # 获取叶子节点数量
    depth = getTreeDepth(myTree)                        # 获取决策树深度
    firstStr = list(myTree.keys())[0]                   # 获取根节点特征标签
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 计算节点中心坐标
    plotMidText(cntrPt, parentPt, nodeTxt)              # 绘制节点文本
    plotNode(firstStr, cntrPt, parentPt, decisonNode)   # 绘制节点
    secondDict = myTree[firstStr]                       # 获取根节点对应的子树
    plotTree.yOff -= 1.0 / plotTree.totalD              # 更新y坐标
    for key in secondDict.keys():                       # 遍历子树的所有特征值
        if isinstance(secondDict[key], dict):           # 如果子树是字典，递归调用 plotTree 函数绘制子树
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff += 1.0 / plotTree.totalW      # 更新x坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)  # 绘制叶子节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  # 绘制叶子节点文本
    plotTree.yOff += 1.0 / plotTree.totalD              # 更新y坐标


def createPlot(inTree):
    """
    创建绘图
    :param inTree: 决策树
    """
    fig = plt.figure(1, facecolor='white')              # 创建绘图窗口
    fig.clf()                                          # 清空绘图窗口
    axprops = dict(xticks=[], yticks=[], frameon=False)  # 设置坐标轴属性
    createPlot.ax1 = plt.subplot(111, frameon=False)  # 创建子图
    plotTree.totalW = float(getNumLeafs(inTree))       # 获取叶子节点数量
    plotTree.totalD = float(getTreeDepth(inTree))      # 获取决策树深度
    plotTree.xOff = -0.5 / plotTree.totalW             # 初始化x坐标
    plotTree.yOff = 1.0                                # 初始化y坐标
    plotTree(inTree, (0.5, 1.0), '')                   # 绘制决策树
    plt.show()                                         # 显示绘图
    createPlot.ax1.cla()                              # 清空绘图窗口    


def retrieveTree(i):
    """
    从文件中读取决策树
    :param i: 决策树索引
    :return: 决策树
    """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: {'head': {0: 'no', 1: 'yes'}}}}}}]
    return listOfTrees[i]                              # 返回指定索引的决策树
