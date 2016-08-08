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

    dataset = mat(dataset)
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
PS: the method of distance will most influence the performance!
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
        centroids[:,i] = minV + random.rand(k,1) * (maxV - minV)

    return centroids

def preview(sample):
    n,m = sample.shape
    ax=plot.figure()
    f1=ax.add_subplot(111)
    f1.plot(sample,'ko')
    plot.show()

def previewKMean(sample,assment,centroids):
    n,m = sample.shape
    ax=plot.figure()
    f1=ax.add_subplot(111)

    k=centroids.shape[0]

    #plot the centroids first
    f1.plot(centroids[:,0],centroids[:,1],'k+',markersize=40)

    colorlist=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markerlist=['+','x','o']

    #plot each cluster with different color
    for i in range(k):
        cluster = sample[nonzero(assment[:,0]==i)[0]]
        f1.plot(cluster[:,0],cluster[:,1], colorlist[i]+'o')

    plot.show()

def randData(m,n,minv=0.0,maxv=100.0):
    dataset=mat(zeros((m,n)))
    dataset=minv + random.rand(m,n)*(maxv-minv)
    return dataset

def randKData(m,n,k,minv=0,maxv=100):
    dataset=[]

    delta=int((maxv-minv)/k)
    for d in range(minv,maxv,delta):
        tmp=randData(int(m/k),n,d,d+delta)
        print tmp

        dataset.extend(tmp)

    dataset=array(dataset)

    return dataset

def KMeans(dataset,k):
    centroids=randCent(dataset,k)
    print centroids
    changed=True
    datanum=len(dataset)
    m,n=dataset.shape
    assment=mat(zeros((m,2)))

    while changed:
        #stop if no change
        changed=False
        print centroids

        for did in range(datanum):
            d=dataset[did]
            minus= inf
            nearid=0

            #find the nearest centroid
            for cid in range(k):
                c=centroids[cid]
                dis=distEuclid(d,c)
                if(dis<minus):
                    minus=dis
                    nearid=cid

            #check if changed
            if(assment[did,0]!=nearid):
                changed=True; #go on if changed

            assment[did]=nearid,minus**2

        #adjust all centroids
        for cid in range(k):
            ids = nonzero(assment[:,0].A==cid)[0]
            print ids
            cluster= dataset[ids]
            centroids[cid,:]= mean(cluster, axis=0)

    return assment, centroids

if __name__ == "__main__":
    #dataset=randKData(50,2,2)
    #print dataset
    #exit(0)

    dataset = file2matrix("KMeans-input.txt")
    print dataset[0]

    #test with 3 centroids
    centroids=randCent(dataset,2)
    print centroids

    #preview(dataset)

    assment,centroids=KMeans(dataset,2)
    previewKMean(dataset,assment,centroids)
