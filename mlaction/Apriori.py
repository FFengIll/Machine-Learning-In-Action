import pprint

from numpy import *

name_map = {}
id_map = {}
map_id = 0


def translate(src):
    global name_map, map_id, id_map

    dest = []
    for i in src:
        if i not in name_map:
            name_map[i] = map_id
            id_map[map_id] = i
            map_id += 1

        dest.append(name_map[i])

    return dest


def re_translate(src):
    global name_map, map_id

    dest = []
    for i in src:
        if i not in name_map:
            name_map[i] = map_id
            map_id += 1

        dest.append(name_map[i])

    return dest


def loadData(filename):
    dataset = []

    with open(filename) as fp:
        for line in fp:
            line = line.strip()

            # drop the first data
            subset = line.split("\t")[1:]

            # translate the data to number
            subset = translate(subset)
            dataset.append(subset)

    return dataset


def createC1(dataset):
    """
    will create the candidates with only 1 element, aka C1, conbination 1.
    this will become the basic of our later work.
    :param dataset:
    :return:
    """

    C1 = []
    for subset in dataset:
        for item in subset:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport=0.5):
    """
    will scan the dataset, and analysis the frequency of Ck.
    of course we will drop the set under the min support, which will help to cut branch.
    :param D:
    :param Ck:
    :param minSupport:
    :return:
    """

    frequency = dict()
    # scan dataset and get the frequency of each c in Ck
    for subset in D:
        for c in Ck:
            if c.issubset(subset):
                if c in frequency:
                    frequency[c] += 1
                else:
                    frequency[c] = 1

    total = float(len(D))
    supportDeg = {}
    res = []

    # compute the support degree, and drop(not store) the unsupport candidates
    for c in Ck:
        support = frequency[c] / total
        if support >= minSupport:
            # res.insert(0,c)
            res.append(c)
        # we store the support degree for later work: calculate the confidence
        supportDeg[c] = support

    return res, supportDeg


def aprioriGen(Lk, k):
    """
    Be careful, during apriori, we will use the Layer to generate another Layer, aka Layer K.
    we only use the layer to generate k! We should confirm the network structure.
    e.g. 01, 02, 03, 12, 13, 23 will generate only 012, 023, 013, 123 but no 1234 (why? see the next)
    so we are using Ck, aka the combination k - in code, we use the prefix to confirm.
    :param Lk:
    :param k:
    :return:
    """

    res = []
    lenlk = len(Lk)
    for i in range(lenlk):
        for j in range(i + 1, lenlk):
            '''
            we choose k-2 of same elements, and the other 2 will be different.
            so then we can get the set of k and confirm no duplication generation.
            PS: each time we call the function, we can keep doing the same work (set has no order),
            so we can always complete the generation by meeting the rules without duplication.
            '''
            La = list(Lk[i])[:k - 2]
            Lb = list(Lk[j])[:k - 2]
            # the sort will help in compare
            La.sort()
            Lb.sort()
            if La == Lb:
                res.append(Lk[i] | Lk[j])
    return res


def apriori(dataset, minSupport=0.5):
    """
    Apriori: is a algorithm to analysis the frequency of each conbination meet the support degree;
    and then extract the rules from the combination according to the confidence degree.
    PS1: the low support degree means the conbination of supreset won't meet either.
    PS2: the low confidence degree means the rule of subset won't meet either.
    so we can cut the branch and increase the speed.
    :param dataset:
    :param minSupport:
    :return:
    """
    C1 = createC1(dataset)
    print("C1: {}".format(C1))
    L1, support = scanD(dataset, C1, 0.5)
    print(L1)
    print(support)

    L = [L1]
    k = 2  # we will generate from C2

    # use the newest L
    while len(L[-1]) > 0:
        Ck = aprioriGen(L[-1], k)
        print("C{}: {}".format(k, Ck))

        Lk, supK = scanD(dataset, Ck, minSupport)
        support.update(supK)
        L.append(Lk)
        k += 1
    return L, support


def calcConf(conseqGroup, freqSet, support, rules, minConf=0.5):
    prunedGroup = []
    for conseq in conseqGroup:
        conf = support[freqSet] / support[freqSet - conseq]
        if conf > minConf:
            print(freqSet - conseq, "-->", conseq, "confidence:", conf)
            rules.append((freqSet - conseq, conseq, conf))
            # only the confident node can be used later, aka pruned of the tree
            prunedGroup.append(conseq)
    return prunedGroup


def getRulesFromSet(conseqGroup, freqSet, support, rules, minConf=0.5):
    n = len(freqSet)
    m = len(conseqGroup[0])

    '''
    check the rules at first, and if possible, then extend the consequence and generate new rules
    '''
    tmpGroup = calcConf(conseqGroup, freqSet, support, rules, minConf)
    # m is the consequence, and we must have at least 1 as prefix, then we need more than (m+1)
    if n > m + 1:
        tmpGroup = aprioriGen(tmpGroup, m + 1)
        if len(tmpGroup) > 1:
            getRulesFromSet(tmpGroup, freqSet, support, rules, minConf)
    return

    '''
    here is another process without the 1-element consequence (they can be combined and ignored), this is a standard process in the book.
    we have a Theorem:
    If a rule X -> Y-X does not meet confidence, then  X' -> Y-X' can not meet either, where X' is a subset of X.
    So we can use the theorem to do prune.
    '''
    if n > m + 1:
        tmpGroup = aprioriGen(conseqGroup, m + 1)
        tmpGroup = calcConf(tmpGroup, freqSet, support, rules, minConf)
        if len(tmpGroup) > 1:
            getRulesFromSet(tmpGroup, freqSet, support, rules, minConf)


def createRules(L, support, minConf):
    """
    an entry to create rules from support and frequency set
    :param L:
    :param support:
    :param minConf:
    :return:
    """
    rules = []
    # of course we need more than 1 element
    for i in range(1, len(L)):
        for freqSet in L[i]:
            conseqGroup = [frozenset([item]) for item in freqSet]
            if i > 1:
                # for more than 2, we have to prune and combine
                getRulesFromSet(conseqGroup, freqSet, support, rules, minConf)
            else:
                # for 2 elements, we can directly calculate
                calcConf(conseqGroup, freqSet, support, rules, minConf)
    return rules


def index2name(rules):
    res = []
    for left, right, conf in rules:
        left = [id_map[i] for i in left]
        right = [id_map[i] for i in right]
        res.append((left, right, conf))
    res.sort(key=lambda x: x[2])
    return res


if __name__ == "__main__":
    filename = "testcase/input/Apriori.txt"
    dataset = loadData(filename)
    print(dataset)

    minSupport = 0.4
    minConf = 0.7

    L, support = apriori(dataset, minSupport)
    print("complete apriori preparation")
    for Lk in L:
        print(Lk)

    for s in support:
        print(s, support[s])

    rules = createRules(L, support, minConf)
    rules = index2name(rules)
    pprint.pprint(rules, indent=4)
