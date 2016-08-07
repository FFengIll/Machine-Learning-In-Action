import sys,os
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot

def file2matrix(filename):
    input=open(filename)
    classvec=[]
    dataset=[]
    
    for line in input:
        line=line.strip()
        datalist = line.split('\t')
        
        datalist[1]=float(datalist[1])
        datalist[0]=float(datalist[0])

        #last data is the class
        dataset.append(datalist)
        
    dataset = array(dataset)
    return dataset

def autonorm(sample):
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
    

'''
A method to calculate Euclidean distance.
Use matrix means batch calculation.
'''    
def distEuclid(veca,vecb):
    delta=(veca-vecb)
    powerVal=power(delta,2)
    res= sqrt(sum(powerVal))
    return res
    
'''
Randomly generate the centroids for the data set.
But we have to limit the random in the valid range!
So check the min and max.
'''    
def randCent(dataset,k):
    m,n=shape(dataset)
    #create the zeros with k node
    centroids= mat (zeros((k,n)))
    
    #range in min and max
    for i in range(n):
        minV = min(dataset[:,i])
        maxV = max(dataset[:,i])
        #set this axis using random list
        centroids[:,i] = minV + (maxV - minV) * random.rand(k,1)

    return centroids
        
def preview(sample):
    n,m = sample.shape
    fig=plot.figure()
    f1=fig.add_subplot(111)
    f1.scatter(sample[:,0], sample[:,1])
    plot.show()

def randData(num,k,maxv=100,minv=0):
    dataset=mat(zeros((num,k)))
    dataset=random.rand(num,k)*(maxv-minv)
    return dataset

if __name__ == "__main__":
    '''
    dataset=randData(200,2) 
    print dataset
    exit(0)
    '''
    dataset = file2matrix("KMeans-input.txt")
    print dataset[0]
    
    #test with 3 centroids
    centroids=randCent(dataset,3)
    print centroids
    
    #dataset, deltas, minvals=autonorm(dataset)
    #print dataset
    
    #preview(dataset)