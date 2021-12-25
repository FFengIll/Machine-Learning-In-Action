import os
import sys
import numpy as np
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot

'''
We define the class to meat OOP,
but we only use a diction to store attr. in our application. (easy way)
'''


class Node():
    def __init__(self, feat, val, left, right):
        self.left = left
        self.right = right
        self.val = val
        self.feat = feat


def loadData(filename):
    input = open(filename)
    classvec = []
    dataset = []

    for line in input:
        line = line.strip().split('\t')

        # convert to float by map method
        datalist = list(map(float, line))

        # last data is the class
        dataset.append(datalist)

    return dataset


def binSplit(dataset, feature, value):
    ids = nonzero(dataset[:, feature] > value)[0]
    s1 = dataset[ids, :]
    ids = nonzero(dataset[:, feature] <= value)[0]
    s2 = dataset[ids, :]
    return s1, s2


def regLeaf(dataset):
    return mean(dataset[:, -1])


'''
calculate the var of the result, 
return total variance (aka multiple m, the number of sample)    
'''


def regError(dataset):
    return var(dataset[:, -1]) * shape(dataset)[0]


'''
Get the feasible values and try to split the dataset.
Calculate the rss error and get the best one split.
'''


def chooseBestSplit(dataset, leafMethod=regLeaf, errMethod=regError, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]

    tmp = dataset[:, -1].T.tolist()[0]
    differSet = set(tmp)
    # no different elements, aka all same
    if len(differSet) == 1:
        return None, leafMethod(dataset)

    m, n = shape(dataset)
    S = errMethod(dataset)

    bestS = inf
    bestIndex = 0
    bestVal = 0
    bestLeft = None
    bestRight = None

    '''
    Try to split and choose.
    Be careful, the last one is result but variables
    '''
    for featId in range(n - 1):
        tmp = dataset[:, featId].T.tolist()[0]
        splitSet = set(tmp)
        for splitVal in splitSet:
            s1, s2 = binSplit(dataset, featId, splitVal)
            # Optimization: if not feat the min sample number, aka tolerate N, drop it
            if shape(s1)[0] < tolN or shape(s2)[0] < tolN:
                continue
            # update best
            err1 = errMethod(s1)
            err2 = errMethod(s2)
            err = err1 + err2
            if err < bestS:
                bestS = err
                bestIndex = featId
                bestVal = splitVal
                bestLeft = s1
                bestRight = s2

    # Optimization: if not meat the tolerate S, drop it, aka the result is not remarkable
    if S - bestS < tolS:
        return None, leafMethod(dataset)

    # may no better split (or split is too small, see the Optimization)
    if bestLeft is None or bestRight is None:
        return None, leafMethod(dataset)

    return bestIndex, bestVal


def CART(dataset, leafMethod=regLeaf, errMethod=regError, ops=(1, 4)):
    # choose the best split
    feat, val = chooseBestSplit(dataset, leafMethod, errMethod, ops)

    # if no more feasible feature, just stop (in recursion)
    if feat is None:
        return val

    # update info, we use dict to store but define a class
    tree = {}
    tree['Feat'] = feat
    tree['Value'] = val

    # split and then get the sub tree (or node)
    s1, s2 = binSplit(dataset, feat, val)
    tree['left'] = CART(s1, leafMethod, errMethod, ops)
    tree['right'] = CART(s2, leafMethod, errMethod, ops)

    return tree


def viewTree(node, allsplit, tab=0):
    if isinstance(node, dict):
        feat = node['Feat']
        val = node['Value']
        allsplit.append((feat, val))

        print("\t" * tab, end=' ')
        print("Feat:%d Value:%f" % (feat, val))

        viewTree(node['left'], allsplit, tab + 1)
        viewTree(node['right'], allsplit, tab + 1)

    else:
        print("\t" * tab, end=' ')
        print(node)


def preview(dataset, cartTree):
    allsplit = []
    viewTree(tree, allsplit)
    print(allsplit)

    plot.plot(dataset[:, 0:-1], dataset[:, -1], 'g+')
    maxval = max(dataset[:, -1])
    minval = min(dataset[:, -1])
    for feat, val in allsplit:
        plot.vlines(val, minval, maxval)

    plot.show()


if __name__ == "__main__":
    '''
    #unit test input
    dataset = [[1,1,1] ,[1,0,1] ,[1,0,1]]
    dataset = matrix(dataset)
    print dataset
    #unit test
    print binSplit(dataset,1,0.5)
    '''

    # here we give 2 dataset for demo (2-dimensional data)
    dataset = loadData("../testcase/CART-input.txt")
    dataset = matrix(dataset)
    tree = CART(dataset, regLeaf, regError, (0.5, 4))
    preview(dataset, tree)

    dataset = loadData("../testcase/CART-input2.txt")
    dataset = matrix(dataset)
    tree = CART(dataset, regLeaf, regError, (1, 4))
    preview(dataset, tree)
