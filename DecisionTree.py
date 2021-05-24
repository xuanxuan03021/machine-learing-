import math
import numpy as np
from matplotlib import pyplot as plt
#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    classLable=dataSet[:,-1]
    classCateg=set(classLable)#有多少种类别
    numEntries=len(classLable)
    lableCounts={}
    for lable in classCateg:
       # print(lable)
       # print(np.where(classLable == lable))#返回结果(array([0, 1]),)，所以需要【0】
        lableCounts[lable] = len((np.where(classLable == lable)[0]))
   # print(lableCounts)
    shannonEnt=0
    for key in lableCounts:
        prob=float(lableCounts[key])/numEntries
        shannonEnt-=prob*math.log(prob,2)
    return shannonEnt

def CreatdataSet():
    dataSet=np.array([[1,1,"yes"],
             [1,1,"yes"],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']])
    lables=["no surfacing",'flippers']
    return  dataSet,lables

#划分数据集
def splitDataSet(dataSet,axis,value):

    retDataSet=[]
    for data in dataSet:
       # print(int(data[axis]) == int(value))#注意value可能传过来是string，所以需要转换成INT
        if int(data[axis]) == int(value):
            reducedFeatVec=np.append(data[:axis],data[axis+1:])#将用于分类的变量删除
            # print("*****")
            # print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    retDataSet=np.array(retDataSet)

    return retDataSet


def chooseBestFeatureTopSplit(dataSet):
    numFeatures=np.size(dataSet,axis=1)-1
   # print(numFeatures)
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0
    bestFeature=-1
    for i in range(numFeatures):
        featureValue=set(dataSet[:,i])
       # print(featureValue)
        newEntropy=0
        for value in featureValue:
            retDataSet=splitDataSet(dataSet,i,value)
            prob=np.size(retDataSet,axis=0)/np.size(dataSet,axis=0)
            # print("***")
            # print(retDataSet)
            newEntropy+=prob*calcShannonEnt(retDataSet)
        infoGain=baseEntropy-newEntropy
        if bestInfoGain<infoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorCnt(classList):
    classListLable=set(classList)
    lableCounts={}
    for lable in classListLable:
        lableCounts[lable] = len((np.where(classList == lable)[0]))
    sortedClassCount = sorted(lableCounts.items(), key=lambda x: x[1], reverse=False)
    return sortedClassCount[0][0]

def creatTree(dataSet,labels):
    classList=dataSet[:,-1]
    if len(np.where(classList==classList[0])[0])==np.size(classList,axis=0):
        return classList[0]
    if np.size(dataSet,axis=1)==1:
        return majorCnt(classList)
    bestFeature=chooseBestFeatureTopSplit(dataSet)
    print("Beat Feature is ",bestFeature)
    beatFeatureLable=labels[bestFeature]
    myTree={beatFeatureLable:{}}
    del labels[bestFeature]
    featureVal=set(dataSet[:,bestFeature])
    print("the value of best feature ",featureVal)
    for value in featureVal:
        subLables=labels[:]
        myTree[beatFeatureLable][value]=creatTree(splitDataSet(dataSet,bestFeature,value),subLables)

    return myTree

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args=dict(arrowstyle="<-")
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)
#获取叶子结点的数目
def getNumLeafs(myTree):
    numLeafs=0

    print(myTree.keys())
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            numLeafs+=getNumLeafs(secondDict[key])
        else: numLeafs+=1
    return numLeafs
#获取树的层数
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth:#比较各个枝谁长谁短
           maxDepth=thisDepth
    return maxDepth

def ployMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):

    leafNode=dict(boxstyle="round4",fc="0.8")
    arrow_args=dict(arrowstyle="<-")
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    ployMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,plotTree.decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            ployMidText((plotTree.xOff,plotTree.yOff), cntrPt, str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD



def createPlot(inTree):
    plotTree.decisionNode=dict(boxstyle="sawtooth",fc="0.8")#变成字典形式{'boxstyle': 'sawtooth', 'fc': '0.8'}
    leafNode=dict(boxstyle="round4",fc="0.8")
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


def main():
    dataSet,lables=CreatdataSet()

    print(chooseBestFeatureTopSplit(dataSet))

    myTree=creatTree(dataSet,lables)
    print(myTree)
    print("leaves",getNumLeafs(myTree))
    print(myTree)
    print("depth",getTreeDepth(myTree))
    createPlot(myTree)

#使用文本注解绘制树节点


if __name__ == '__main__':
    main()