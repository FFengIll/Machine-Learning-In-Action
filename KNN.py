import sys,os
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot

def loadData(filename):
    input=open(filename)
    classvec=[]
    dataset=[]
    
    for line in input:
        line=line.strip()
        datalist = line.split('\t')
        
        #last data is the class
        data = [float(v) for v in datalist[1:-1]]
        type = datalist[-1]
        
        dataset.append(data)
        classvec.append(type)
        
    dataset = array(dataset)
    classvec = array(classvec)
    return dataset, classvec

def normalize(sample):    
    #arg = 0 means get min or max in collumn
    minvals= sample.min(0)
    maxvals=sample.max(0)
    print minvals
    print maxvals
    deltas = maxvals -  minvals

    #data to float to avoid integer division
    deltas = deltas*1.0
      
    n,m=sample.shape
    normSample=zeros((n,m))
    
    normSample= sample- tile(minvals, (n,1))
    print normSample
    normSample=normSample/tile(deltas, (n,1))
    print normSample
    
    return normSample, deltas, minvals
    
def preview(sample):
    n,m = sample.shape
    fig=plot.figure()
    f1=fig.add_subplot(111)
    f1.scatter(sample[:,0], sample[:,1])
    plot.show()

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

def classify1(input, sample, label, k=3):
    pass

if __name__ == "__main__":
    matrix, classvec = loadData("KNN-input.txt")
    print matrix
    print classvec
    
    '''
    We need to do normalization, 
    the value distribution of each feature will influence the distance seriously
    '''
    matrix, deltas, minvals=normalize(matrix)
    print matrix
    #preview(matrix)
    
    classify0(matrix[-1], matrix, classvec)