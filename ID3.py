import sys,os
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot
from math import log

def unify(data):
    for i in range(data.__len__()):
        if data[i]=='Y':
            data[i]=1
        elif data[i]=='N':
            data[i] = 0
    return data     

def file2matrix(filename):
    input=open(filename)
    classvec=[]
    dataset=[]
    
    for line in input:
        line=line.strip()
        datalist = line.split('\t')

        datalist = unify(datalist)
        
        #last data is the class
        dataset.append(datalist[1:-1])
        classvec.append(datalist[-1])
        
    dataset = array(dataset)
    classvec = array(classvec)
    return dataset, classvec

def calcShannonEnt(dataset):
    n, m = dataset.shape
    labelCount={}
    for data in dataset:
        curlabel=data[-1]
        labelCount[curlabel] = labelCount.get(curlabel,0) + 1

    shannonEnt=0.0
    for key in labelCount:
        prob= float(labelCount[key])/n
        mid = prob * log(prob, 2)
        shannonEnt -= mid 

    return shannonEnt 

def classify0(input, sample, label, k=3):
    #getDistance(sample[0], sample[1])

    #shape means it is a N * M matrix
    dataSetSize= sample.shape[0]
    diffMat= tile(input, (dataSetSize,1)) - sample
    print diffMat
    sqDiffMat= diffMat**2
    sqDistance= sqDiffMat.sum(axis=1)
    print sqDistance
    distanceM= sqDistance**0.5
    print distanceM
    
    #internal sort
    #distanceM.sort()
    print distanceM
    #return the index of the result of sort
    sortDisIndex= distanceM.argsort()
    print sortDisIndex
    
    classcount={}
    for i in range(k):
        #use sort index to get label
        unlabel= label[ sortDisIndex[i] ]
        #use dict to store the count of label
        classcount[unlabel]=classcount.get(unlabel,0) + 1
    
    #choose the best  
    print classcount

def splitDataSet(dataset, axis, value):
    retSet=[]
    for data in dataset:
        #match axis value first
        if data[axis]==value:
            #split the axis data
            mid= data[:axis]
            mid.extend(data[axis+1:])
            retSet.append (mid)
    return retSet

def chooseBestSplit(dataset):
    #feature number except the result one
    featureNum = len(dataset[0])
    featureNum -= 1
    
    #calc base Shannon Entropy
    baseEnt = calcShannonEnt(dataset)
    bestFeature = -1
    bestGain = 0.0

    for i in range(featureNum):
        featureList = [ data[i] for data in dataset]
        uniFeature = set (featureList)
        print uniFeature


if __name__=="__main__":
    sample, classvec = file2matrix("ID3-input.txt")
    print sample
    print classvec
    #preview(matrix)
    #classify0(sample[-1], sample, classvec)
    #print sample**2
    print calcShannonEnt(sample)
    chooseBestSplit(sample)
    