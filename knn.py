# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:47:08 2017

@author: Administrator
"""
import numpy as np
import operator
from os import listdir

'''创建数据集'''
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


'''分类算法'''
def classify0(inX,dataSet,labels,k):
    '''计算距离'''
    datasetSize=dataSet.shape[0]
    diffmat=np.tile(inX,(datasetSize,1))-dataSet
    sqlmat=diffmat**2
    distancemat=sqlmat.sum(axis=1)
    distance=distancemat**0.5
    '''按照距离递增次序排列,返回的是索引值'''
    distanceindice=distance.argsort()
    classcount={}
    '''选取前k个，并判断分类次数最多的是哪个类别并返回'''
    for i in range(k):
        votelabel=labels[distanceindice[i]]
        classcount[votelabel]=classcount.get(votelabel,0)+1
    classcountsort=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return classcountsort[0][0]


'''处理约会文件数据'''
def file2matrix(filename):
    '''传入的是文件名，得到特征向量和类标签向量'''
    with open(filename) as f:
        arraylines=f.readlines()
        size=len(arraylines)
    matrix=np.zeros((size,3))
    index=0
    resultvector=[]
    for line in arraylines:
        newline=line.strip()
        linelist=newline.split('\t')
        matrix[index,:]=linelist[0:3]
        resultvector.append(int(linelist[-1]))
        index+=1
    return matrix,resultvector


'''归一化特征值'''
def autoNorm(dataset):
    maxvalue=dataset.max(0)  #返回每一列的最大值
    minvalue=dataset.min(0)  #返回每一列的最小值
    ranges=maxvalue-minvalue
    normdataset=np.zeros(np.shape(dataset))
    m=dataset.shape[0]
    minval=np.tile(minvalue,(m,1))
    normdataset=dataset-minval
    normdataset=normdataset/np.tile(ranges,(m,1))
    return normdataset,minvalue,ranges


'''针对约会网站的测试代码'''
def datingclassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('E:\\test\\datingTestSet2.txt')
    datingData,ranges,minval=autoNorm(datingDataMat)
    m=datingData.shape[0]
    numtestVecs=int(hoRatio*m)
    errorcount=0
    for i in range(numtestVecs):
        classifierresult=classify0(datingData[i,:],datingData[numtestVecs:m,:],datingLabels[numtestVecs:m],3)
        print('the classifierresult is:%d,the real result is:%d' %(classifierresult,datingLabels[i]))
        if(classifierresult!=datingLabels[i]):
            print('the result is error')
            errorcount+=1
    print('the error ratio is:%f' %(errorcount/numtestVecs))
    
    
'''约会网站预测函数（真正开始使用算法）'''
def classifyPerson():
    resultList=['not at all','a small doses','a large doses']
    game=float(input('花在打电子游戏上的时间:'))
    mile=float(input('飞行距离：'))
    icecream=float(input('花在吃冰激凌的时间:'))
    inArr=np.array([mile,game,icecream])
    dataSet,labels=file2matrix('E:\\test\\datingTestSet2.txt')
    dataset,minvalue,ranges=autoNorm(dataSet)
    classied=classify0((inArr-minvalue)/ranges,dataset,labels,3)
    print('the result is:',resultList[classied-1])
    
    
'''将手写数据集图像转换为向量'''    
def img2vector(filename):
    returnvector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr=fr.readline()
        for j in range(32):
            returnvector[0,32*i+j]=int(linestr[j])
    return returnvector
        
'''手写数字识别系统的测试代码'''
def handwritingClassTest():
    trainnigList=listdir('E:\\machinelearninginaction\\Ch02\\trainingDigits')
    m=len(trainnigList)
    trainningMat=np.zeros((m,1024))
    hwlabels=[]
    for i in range(m):
        filename=trainnigList[i]
        trainningMat[i,:]=img2vector('E:\\machinelearninginaction\\Ch02\\trainingDigits\\%s' %(filename))
        label=filename.split('.')[0]
        hwlabel=int(label.split('_')[0])
        hwlabels.append(hwlabel)
    testList=listdir('E:\\machinelearninginaction\\Ch02\\testDigits')
    n=len(testList)
    errorcount=0
    for j in range(n):
        filename2=testList[j]
        testingdata=img2vector('E:\\machinelearninginaction\\Ch02\\testDigits\\%s' %(filename2))
        testlabel=filename2.split('.')[0]
        hwtestlabel=int(testlabel.split('_')[0])
        result=classify0(testingdata,trainningMat,hwlabels,3)
        print('the predicted result is:%d,the real result is:%d' %(result,hwtestlabel))
        if(result!=hwtestlabel):
            print('the prediction is error')
            errorcount+=1
    print('the error total number is:',errorcount)
    print('the error rate is:',errorcount/n)
    
'''主函数'''
if __name__ == '__main__':
    handwritingClassTest()
