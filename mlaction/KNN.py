import sys
import os
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot
from matplotlib import colors
from collections import defaultdict


def loadData(filename):
    classvec = []
    dataset = []
    with open(filename) as fd:
        for line in fd:
            line = line.strip()
            datalist = line.split('\t')

            # last data is the class
            data = [float(v) for v in datalist[1:-1]]
            type = datalist[-1]

            dataset.append(data)
            classvec.append(type)

        dataset = array(dataset)
        classvec = array(classvec)
    return dataset, classvec


def get_bound(sample):
    # arg = 0 means get min or max in collumn
    minvals = sample.min(0)
    maxvals = sample.max(0)
    print(minvals)
    print(maxvals)
    return maxvals, minvals


def normalize(sample):
    maxvals, minvals = get_bound(sample)
    deltas = maxvals - minvals

    # data to float to avoid integer division
    deltas = deltas * 1.0

    n, m = sample.shape
    normSample = zeros((n, m))

    # process by element
    normSample = sample - tile(minvals, (n, 1))
    normSample = normSample / tile(deltas, (n, 1))

    return normSample, deltas, minvals


def get_distance(unknown, known):
    dataSetSize = known.shape[0]
    diffMat = tile(unknown, (dataSetSize, 1)) - known
    print(diffMat)
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    print(sqDistance)
    distanceM = sqDistance ** 0.5
    print(distanceM)

    return distanceM


def classify0(unknown, known, label, k=3):
    """
    get the most possilbe k point by distance
    then do vote
    :param unknown:
    :param known:
    :param label:
    :param k:
    :return:
    """

    # shape means it is a N * M matrix
    distanceM = get_distance(unknown, known)

    print(distanceM)
    # return the index of the result of sort
    sortDisIndex = distanceM.argsort()
    print(sortDisIndex)

    classcount = defaultdict(lambda: 0)
    for i in range(k):
        # use sort index to get label
        unlabel = label[sortDisIndex[i]]
        # use dict to store the count of label
        classcount[unlabel] += 1

    # choose the best
    print(classcount)


def classify1(input, sample, label, k=3):
    pass


COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def preview(sample, label=None):
    label_set = list(set(label))
    label_color = {}

    id = 0
    for item in label:
        if item in label_color:
            continue
        label_color[item] = COLOR_LIST[id]
        id += 1

    n, m = sample.shape
    fig = plot.figure()

    x, y = sample[:, 0], sample[:, 1]
    color = [label_color.get(i) for i in label]
    sub1 = fig.add_subplot(111)
    sub1.scatter(x, y, c=color)

    plot.show()


if __name__ == "__main__":
    matrix, classvec = loadData("../testcase/KNN-input.txt")
    print(matrix)
    print(classvec)

    '''
    We need to do normalization, 
    The value distribution of each feature will influence the distance seriously.
    Of course, all data including unknown should concluded!
    '''
    norm_matrix, deltas, minvals = normalize(matrix)
    print(norm_matrix)
    preview(norm_matrix, classvec)

    # split known and unknown by label
    classify0(norm_matrix[-1], norm_matrix[0:-1], classvec)
