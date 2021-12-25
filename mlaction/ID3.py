from math import log

from numpy import *


def loadData(filename):
    input = open(filename)
    classvec = []
    dataset = []

    for line in input:
        line = line.strip()
        datalist = line.split('\t')

        # last data is the class
        data = [v for v in datalist[1:-1]]
        type = datalist[-1]

        dataset.append(data)
        classvec.append(type)

    dataset = dataset
    classvec = classvec
    return dataset, classvec


def calcShannonEnt(dataset):
    n = len(dataset)
    labelCount = {}
    for data in dataset:
        curlabel = data[-1]
        labelCount[curlabel] = labelCount.get(curlabel, 0) + 1

    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / n
        mid = prob * log(prob, 2)
        shannonEnt += -mid

    return shannonEnt


def classify0(input, sample, label, k=3):
    pass


def splitDataSet(dataset, axis, value):
    retSet = []
    for data in dataset:
        # match axis value first
        if data[axis] == value:
            # split the axis data
            newdata = data[:axis] + data[axis + 1:]
            # newdata.extend(  )
            retSet.append(newdata)
    return retSet


def chooseBestSplit(dataset):
    """
    choose the best feature to split the dataset.
    this feature will give the best entropy gain which means data classification is clear
    :param dataset:
    :return:
    """

    # feature number except the result one
    featureNum = len(dataset[0])
    featureNum -= 1

    # calc base Shannon Entropy
    baseEnt = calcShannonEnt(dataset)
    bestFeature = -1
    bestGain = 0.0

    total = float(len(dataset))

    for i in range(featureNum):
        featureList = [data[i] for data in dataset]
        uniFeature = set(featureList)
        print(uniFeature)

        newEntropy = 0.0
        for val in uniFeature:
            subDataset = splitDataSet(dataset, i, val)
            prob = len(subDataset) / total
            newEntropy += prob * calcShannonEnt(subDataset)

        tmpGain = baseEnt - newEntropy
        if tmpGain > bestGain:
            bestGain = tmpGain
            bestFeature = i

    return bestFeature


def preview_tree(index, label):
    pass


if __name__ == "__main__":
    sample, classvec = loadData("testcase/input/ID3.txt")
    print("sample", sample)
    print("classification", classvec)
    # preview(matrix)
    # classify0(sample[-1], sample, classvec)
    # print sample**2
    shannon_ent = calcShannonEnt(sample)
    print(shannon_ent)
    feature_index = chooseBestSplit(sample)
    print('the best feature index:', feature_index)
    preview_tree()
