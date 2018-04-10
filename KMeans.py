import sys
import os
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot


def file2matrix(filename):
    input = open(filename)
    classvec = []
    dataset = []

    for line in input:
        line = line.strip()
        datalist = line.split('\t')

        datalist[1] = float(datalist[1])
        datalist[0] = float(datalist[0])

        # last data is the class
        dataset.append(datalist)

    dataset = mat(dataset)
    return dataset


def autonorm(sample):
    # arg = 0 means get min or max in collumn
    minvals = sample.min(0)
    maxvals = sample.max(0)
    # print minvals
    # print maxvals
    deltas = maxvals - minvals

    # data to float to avoid integer division
    deltas = deltas * 1.0

    n, m = sample.shape
    normSample = zeros((n, m))

    normSample = sample - tile(minvals, (n, 1))
    # print normSample
    normSample = normSample / tile(deltas, (n, 1))
    # print normSample

    return normSample, deltas, minvals


'''
A method to calculate Euclidean distance.
Use matrix means batch calculation.
PS: the method of distance will most influence the performance!
'''


def distEuclid(veca, vecb):
    delta = (veca - vecb)
    powerVal = power(delta, 2)
    res = sqrt(sum(powerVal))
    return res


'''
Randomly generate the centroids for the data set.
But we have to limit the random in the valid range!
So check the min and max.
'''


def randCent(dataset, k):
    m, n = shape(dataset)
    # create the zeros with k node
    centroids = mat(zeros((k, n)))

    #range in min and max
    for i in range(n):
        minV = min(dataset[:, i])
        maxV = max(dataset[:, i])
        # set this axis using random list
        centroids[:, i] = minV + random.rand(k, 1) * (maxV - minV)

    return centroids


def preview(sample):
    n, m = sample.shape
    ax = plot.figure()
    f1 = ax.add_subplot(111)
    f1.plot(sample, 'ko')
    plot.show()


def previewKMean(clusters, centroids):
    ax = plot.figure()
    f1 = ax.add_subplot(111)

    k = centroids.shape[0]

    for c in centroids:
        print c
    print "*" * 10

    # plot the centroids first
    f1.plot(centroids[:, 0], centroids[:, 1], 'k+', markersize=40)

    colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markerlist = ['+', 'x', 'o']

    # plot each cluster with different color
    for i in range(k):
        c = clusters[i]
        f1.plot(c[:, 0], c[:, 1], colorlist[i] + 'o')

    plot.show()


def randData(m, n, minv=0.0, maxv=100.0):
    dataset = mat(zeros((m, n)))
    dataset = minv + random.rand(m, n) * (maxv - minv)
    return dataset


def randKData(m, n, k, minv=0, maxv=100):
    dataset = []

    delta = int((maxv - minv) / k)
    for d in range(minv, maxv, delta):
        tmp = randData(int(m / k), n, d, d + delta)
        print tmp

        dataset.extend(tmp)

    dataset = array(dataset)

    return dataset


'''
after clustering, divide the dataset and then get all clusters details
'''


def getClusters(dataset, assment, centroids):
    clusters = []
    for i in range(len(centroids)):
        c = dataset[nonzero(assment[:, 0] == i)[0]]
        clusters.append(c)
    return clusters


'''
K Means: a algorithm to find out the centroids to stand for the clusters.
in k means, we randomly choose k centroids, and compute out all clusters;
the we have the assment of all clusters, but the centroids may change (we just do update);
continue to do the job until the centroids keep stable (which is the balance point).
'''


def KMeans(dataset, k):
    centroids = randCent(dataset, k)
    # print centroids
    changed = True
    datanum = len(dataset)
    m, n = dataset.shape
    assment = mat(zeros((m, 2)))

    # we will go on until meet the balance aka the stability!
    while changed:
        # stop if no change, which means the centroids are stable
        changed = False
        # print centroids

        for did in range(datanum):
            d = dataset[did]
            minus = inf
            nearid = 0

            # find the nearest centroid, which is the clusters the node belongs to
            for cid in range(k):
                c = centroids[cid]
                dis = distEuclid(d, c)
                if(dis < minus):
                    minus = dis
                    nearid = cid

            # check if changed
            if(assment[did, 0] != nearid):
                changed = True  # go on if changed

            # we store the assment for later work, and distance too
            assment[did] = nearid, minus**2

        # adjust all centroids, because the clusters details changed and so do centroids
        for cid in range(k):
            ids = nonzero(assment[:, 0].A == cid)[0]
            # print ids
            cluster = dataset[ids]
            centroids[cid, :] = mean(cluster, axis=0)
        '''
        #a preview of clustering
        temp_clusters= getClusters(dataset,assment,centroids)
        temp_centroids= centroids
        previewKMean(temp_clusters,temp_centroids)
        '''
    return assment, centroids


'''
SSE: sum of squared error, is a method to measure the performance of the clustering.
the SSE means the distance between the data node and the cluster centroid.
the less SSE, the better clustering.
'''


def SSE(dataset, assment, centroids):
    k = len(centroids)
    distsum = mat(zeros((k, 1)))

    # in assment, list of pair: nearest centroids id, and distance**2 (no need to compute again)
    for cid in range(k):
        ids = nonzero(assment[:, 0].A == cid)[0]
        dists = assment[ids][:, 1]
        distsum[cid] = sum(dists)
    return distsum


'''
back-process:if the performance of the clustering is not good, we can combine the nearest centroids or
we centroids have least increasing on SSE.
so maybe we can use these ways just in the processing
'''


'''
bisecting K means: a algorithm make all nodes as one cluster and do bisecting with the one cluster.
we can continue to do bisecting with one cluster to decrease the SSE until we meet k.
(WRONG!) PS:we have to do k-means with 2 and then compute the SSE in each loop, and choose the one with largest SSE for the next.
PS:we have to do k-means with 2 for each clusters, and calculate current global SSE; then we have the least SSE which means we accept the clustring in this loop.
'''


def biKMeans(dataset, k, distMeas=distEuclid):
    # init data structure
    clusters = [None] * k
    clusters[0] = dataset
    # print clusters
    # raw_input()
    allassment = [None]
    allcentroids = mat(zeros((k, 2)))
    allsse = mat(zeros((k, 1)))

    # do bisecting k means
    for time in range(1, k, 1):
        leastSSE = inf
        bestSSE = []
        bestCentroids = []
        bestAssment = []
        bestCid = -1

        # find the best bisecting
        for i in range(time):
            assment, centroids = KMeans(clusters[i], 2)
            # compute new sse
            sse_s = SSE(clusters[i], assment, centroids)
            sse_new = sum(sse_s)

            # compute others' sse
            sse_old = 0.0
            for j in range(time):
                if(i == j):
                    continue
                #sse_old += SSE(clusters[j],allassment[j],allcentroids[j])
                sse_old += allsse[j]

            sse_sum = sse_new + sse_old
            if sse_sum < leastSSE:
                bestCid = i
                bestSSE = sse_s
                bestCentroids = centroids
                bestAssment = assment
                leastSSE = sse_sum

        # update all data after choosing
        # get the old cluster
        cluster_old = clusters[bestCid]

        # clear old data
        # allsse.remove(allsse[bestCid])
        # clusters.remove(clusters[bestCid])
        # allcentroids.remove(allcentroids[bestCid])
        space = [bestCid, time]

        # update with new data, of course only 2 clusters
        for cid in range(2):
            ids = nonzero(bestAssment[:, 0].A == cid)[0]
            cluster_new = cluster_old[ids]

            # update the data: clear(actually replace) the old, and add(actually fill) the new
            spaceid = space[cid]
            clusters[spaceid] = cluster_new
            allsse[spaceid] = bestSSE[cid]
            allcentroids[spaceid] = bestCentroids[cid]
        pass

        # a preview of clustering

        temp_clusters = clusters[:time + 1]
        temp_centroids = allcentroids[:time + 1]
        previewKMean(temp_clusters, temp_centroids)

    return clusters, allcentroids


def biKMeansAuto(dataset, k=100, distMeas=distEuclid):
    # init data structure
    clusters = [None] * k
    clusters[0] = dataset
    # print clusters
    # raw_input()
    allassment = [None]
    allcentroids = mat(zeros((k, 2)))
    allsse = mat(zeros((k, 1)))

    # do bisecting k means
    for time in range(1, k, 1):
        leastSSE = inf
        bestSSE = []
        bestCentroids = []
        bestAssment = []
        bestCid = -1

        # find the best bisecting
        for i in range(time):
            assment, centroids = KMeans(clusters[i], 2)
            # compute new sse
            sse_s = SSE(clusters[i], assment, centroids)
            sse_new = sum(sse_s)

            # compute others' sse
            sse_old = 0.0
            for j in range(time):
                if(i == j):
                    continue
                #sse_old += SSE(clusters[j],allassment[j],allcentroids[j])
                sse_old += allsse[j]

            sse_sum = sse_new + sse_old
            if sse_sum < leastSSE:
                bestCid = i
                bestSSE = sse_s
                bestCentroids = centroids
                bestAssment = assment
                leastSSE = sse_sum

        # update all data after choosing
        # get the old cluster
        cluster_old = clusters[bestCid]

        # clear old data
        # allsse.remove(allsse[bestCid])
        # clusters.remove(clusters[bestCid])
        # allcentroids.remove(allcentroids[bestCid])
        space = [bestCid, time]

        # update with new data, of course only 2 clusters
        for cid in range(2):
            ids = nonzero(bestAssment[:, 0].A == cid)[0]
            cluster_new = cluster_old[ids]

            # update the data: clear(actually replace) the old, and add(actually fill) the new
            spaceid = space[cid]
            clusters[spaceid] = cluster_new
            allsse[spaceid] = bestSSE[cid]
            allcentroids[spaceid] = bestCentroids[cid]
        pass

        # a preview of clustering
        '''
        temp_clusters= clusters[:time+1]
        temp_centroids= allcentroids[:time+1]
        previewKMean(temp_clusters,temp_centroids)
        '''

    return clusters, allcentroids


if __name__ == "__main__":
    # dataset=randKData(50,2,2)
    # print dataset
    # exit(0)

    dataset = file2matrix("KMeans-input.txt")
    # print dataset[0]

    '''
    #test with 3 centroids
    centroids=randCent(dataset,2)
    print centroids
    '''

    # preview(dataset)

    '''
    #test the normal k means
    assment,centroids=KMeans(dataset,4)
    clusters=getClusters(dataset,assment, centroids)
    '''

    # test the bisecting k means
    k = 4
    clusters, centroids = biKMeans(dataset, 4)
    previewKMean(clusters, centroids)
