from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from random import randint
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





