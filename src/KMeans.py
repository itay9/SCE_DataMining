from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from random import randint
from src.functions import Discretize, getColumnTitles
import pandas as pd

def makeZero(dataSet):
    """

    @param dataSet: list of data
    @return: list of zero in size of dataSet
    """
    arr = []
    size = len(dataSet)
    for i in range(size):
        arr.append(0)
    return arr


def makeCenterList(data):
    """

    @param data: list of CenterPoint for 1D K-Means
    @return: fixed list of center
    """
    newData =[]
    for i in range(len(data)):
        newData.append(int(data[i][0]))
    return newData

def makePoint(data1,data2):
    """

    @param data1: list of X value
    @param data2: list of Y value
    @return: list of XY value
    """
    result = []
    for i in range(len(data1)):
        result.append((data1[i],data2[i]))
    return result




def makeRandomPointList(numOfPoint):
    """
    generate random point 0<=x,y<=100

    @param numOfPoint: number of point to generate
    @return: list of numOfPoint points (x,y)
    """
    data = []
    for i in range(numOfPoint):
        data.append((randint(0,100),(randint(0,100))))
    return data

def makeXYlist(data):
    """
    split list of (x,y) to 2 lists
    @param data: list of cordinates (x,y)
    @return XY list
        [0] - list of x
        [1] - list of y
    """
    xList = []
    yList = []
    for i in range(len(data)):
        xList.append(data[i][0])
        yList.append(data[i][1])
    result = [xList,yList]
    return result

#Test
#make single line K-Mean
"""
data: List[int] = [1,2,3,8,9,10,15,16,17]
kmeans = KMeans(n_clusters=3)
kmeans.fit(makePoint(data,makeZero(data)))
center = kmeans.cluster_centers_
cList = makeCenterList(center)
#print(center)


#make data in single line
plt.plot(data,makeZero(data), 'ro' )
plt.plot(cList,makeZero(cList), 'bo')
#print(makeCenterList(center))
plt.xticks(list(range(20)))
plt.show()
"""

#make 2D K-Mean
"""
pList = makeRandomPointList(20)
kmeans = KMeans(n_clusters=6)
kmeans.fit(pList)
center = kmeans.cluster_centers_
xyList = makeXYlist(pList)
plt.scatter(xyList[0],xyList[1])
cList = makeXYlist(center)
plt.scatter(cList[0],cList[1])
plt.show()
"""


def numericCol(table, structureTextFile):
    structure = pd.read_csv(structureTextFile, sep=" ", names=['type', 'feature', 'data'])
    column = []
    headers = getColumnTitles(table)
    for i in range(structure.shape[0]):
        if 'NUMERIC' in structure.loc[i]['data']:
            column += [headers[i]]
    return column

def single_kMean(data,cluster = 4):
    """
    works for 1D and 2D data list
    @param data: list of data
    @param cluster: number of cluster, default = 4
    @return: list of K-Mean Cluster
    """
    if isinstance(data[0],int):
        dimension = 1
    else:
        dimension = 2
    cList = []
    kmeans = KMeans(n_clusters=cluster)
    if dimension==1:#fix 1D bug
        kmeans.fit(makePoint(data, makeZero(data))) #tranfer to 2D
    else:
        kmeans.fit(data)
    center = kmeans.cluster_centers_  #calc cluster center
    if dimension==1: #fix 1D bug
        cList = makeCenterList(center)
    elif dimension==2:
        cList = center
    return cList

def makeColDict(columns):
    """

    :param columns: dict of columns name and class count
    :return: dict of colName : yes no count
    """
    colDict = {}
    for col in columns:
        colDict[col] = {'yes': 0, 'no': 0}
    return colDict

def incYes(dict,col):
    """

    :param dict: counter dict
    :param col: column name for inc
    :return: adding 1 yo yesClass count
    """
    dict[col]['yes'] +=1

def incNo(dict,col):
    """

        :param dict: counter dict
        :param col: column name for inc
        :return: adding 1 to noClass count
        """
    dict[col]['no'] +=1

#testFull
"""
data: List[int] = [1,2,3,8,9,10,15,16,17]
print(kMean(data,3))
print(kMean(makeRandomPointList(15),4))
"""
def K_Means(train, test,struct):
    """
    check k means for each
    @param train:  cvs file for training the module
    @param test:  cvs file for testing the module
    @param struct: text file of the cvs structure
    @return:
    """
    numOfCluster = (int)
    numOfCluster = 5
    train = Discretize(numOfCluster,train,struct)
    test = Discretize(numOfCluster,test,struct)
    column = numericCol(train,struct) #get column names

    numOfColumn = len(column)
    print("column: ",column)
    print("num of column ", numOfColumn)
    dict = makeColDict(column)
    print(dict)
    incYes(dict,'age')
    print(dict)



structFile= 'Structure.txt'
trainFile='train.csv'
testFile="test.csv"
train = pd.read_csv(trainFile)
test = pd.read_csv(testFile)
K_Means(train,test,structFile)
    






