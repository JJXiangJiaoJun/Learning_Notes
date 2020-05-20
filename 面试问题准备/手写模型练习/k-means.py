import numpy as np
import math
import random

def loadDataSet(fileName):
    with open(fileName,'r') as f:
        total_record = f.readlines()
    dataSet = [record.strip().split('\t') for record in total_record]
    dataSet = [[float(x) for x in record] for record in dataSet]

    return np.array(dataSet);

def distance(vecA,vecB):
    return math.sqrt(sum((vecA-vecB)**2))

def randCent(dataSet,k):
    idx = list(range(0,dataSet.shape[0]))
    random.shuffle(idx)

    center = dataSet[idx[:k],:]

    return center


def K-means(dataSet,k,dist = distance,createCenter = randCent,maxIter = 100000):
    clusterAssign = np.zeros(shape=(dataSet.shape[0],2))
    center = createCenter(dataSet)
    clusterChange = True
    for epoch in range(maxIter):
        if clusterChange == False:
            break;
        clusterChange = False

        for i in range(dataSet.shape[0]):
            minDist = inf
            minIdx = -1
            for j in range(k):
                curDist = dist(dataSet[i],center[j])
                if(curDist<minDist):
                    minDist = curDist
                    minIdx = j
            if clusterAssign[i][0] != minIdx:
                clusterChange = True
            clusterAssign[i,:] = minIdx,minDist**2
        #重新计算聚类中心
        for cent in range(k):
            cur_data = dataSet[clusterAssign[:,0]==k,:]
            center[cent,:] = np.mean(cur_data,axis = 0)
    
    return center,clusterAssign



