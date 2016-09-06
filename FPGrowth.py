import sys,os
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot

def loadData(filename):
    fp = open(filename)
    line = None
    line=fp.readline()
    dataset=[]
    while(line):
        #print line
        line=line.strip()

        #translate the data to number
        subset = line
        #subset= [v for v in line]
        #print subset
        
        dataset.append(subset)

        #next line
        line=fp.readline()
        
    return dataset

class Node():
    def __init__(self,name, freq, parent, children=None):
        self.name=name
        self.freq=freq
        self.parent=parent
        self.link=None
        self.children = {}

    def inc(self, num):
        self.freq += num
        
    def disp(self,tab=0):
        print "\t" * tab,
        print self.name, self.freq
        for key in self.children.keys():
            child = self.children[key]
            child.disp(tab+1)

def updateHeader(item, header, node):
    #if no such node, add to header, else add to link list at the last
    if( header.has_key(item)):
        tmpnode = header[item]
        #find the last one
        while (tmpnode.link != None):
            tmpnode = tmpnode.link
        tmpnode.link = node
    else:
        header[item] = node
            
def updateTree(record, itemHash,root, header):
    curNode = root
    nextNode = None
    for item in record:
        #invalid item
        if itemHash[item] <= 0:
            continue
            
        #find item in chidren, if no such child, create new one
        if curNode.children.has_key(item):
           nextNode = curNode.children[item]
        else:
            #create new node with parent
            nextNode = Node(item, 0, curNode)            
            curNode.children[item] = nextNode
            #we create a new node, then update the header and list
            updateHeader(item, header, nextNode)
        
        #update
        nextNode.inc(1)
        
        #move
        curNode = nextNode
      
def buildTree(dataset, minSup=1):
    #a header point array to help visit node quickly (each head is a list with same item)
    header = {}

    '''
    First Scan: get global frequency
    '''
    hashtable={}
    for record in dataset:
        for item in record:
            #hashtable[item] = hashtable.get(item,0) + 1
            hashtable[item] = hashtable.get(item,0) + dataset[record]
    
    #if the key not meet min support (aka occurrence or frequency), delete it
    for k in hashtable.keys():
        if (hashtable[k] < minSup):
            del(hashtable[k])
    
    #print hashtable;

    #get freqSet including items, #if no item meet, over the flow
    freqset = set(hashtable.keys())
    #print freqset;
    if len(freqset)<=0:
        return None, None

    #build a tree root - use empty symbol
    root = Node("empty root", 0, parent=None)
    
    '''
    Second Scan: now we start to build the tree
    '''
    for record in dataset:
        localD={}
        #get all item in the record, and global frequency
        for item in record:
            localD[item] = hashtable.get(item,0)
         
        #sort item and delete not support (or keep 0)
        orderRec = [v[0] for  v in sorted(localD.items(), key=lambda p:p[1]) ]
        orderRec.reverse()
        print orderRec;
        #print localD;
        
        '''
        Update the tree: find item node and update, or create new item first.
        But be careful, we must do something to make the same subset be a prefix but break.
        (e.g. z->x->r->t->y, z->x->s->t->y is not good enough, we want z->x->y->*, because zxy is common)
        So the item with same support in sort is hard to process. (we may need to calculate support ratio)
        '''
        updateTree(orderRec,localD,root,header)
        #root.disp()

    return root, header


def ancestorNode(node, minSup=1):
    res=[]
    curNode = node.parent
    
    #visit ancestor node until root
    while curNode != None:
        if curNode.freq >= minSup: 
            name = curNode.name
            res.append(name)
           
        curNode = curNode.parent
    
    #print "prefix of %s:%s" %(node.name, res);
    return res
    
def findPrefix(basePat, tree):
    condPats = {}
    node = tree
    while (node != None):
        prefix = ancestorNode(node)
        if( len(prefix) >0):
            condPats[ frozenset(prefix)] = node.freq
        node = node.link
        
    #print "condition pattern of %s:%s" %(tree.name,condPats);
    return condPats
    
def mineTree(tree, header, minSup, prefix, freqItemList):
    #sort form small to large - we wanna scan from the leaf node (low frequency)
    items = [v[0] for v in sorted(header.items(), key = lambda p:p[1])]

    for basePat in items:
        #generate the new freqSet, aka the new prefix (will use later)
        #of course, all basePat is frequent, and it must be parent of prefix in current tree - frequent too
        newFreqSet = prefix.copy()
        newFreqSet.add(basePat)
        newPrefix = newFreqSet
        freqItemList.append(newFreqSet)
        
        '''
        Go to find the condition patterns, and use them to build a new tree - FP conditional tree.
        This tree will re-grow according to the new statistics result -
        we can analysis all condPats, aka all prefixes, and output a new FP tree which will think of all prefix to find the frequency.
        (e.g. we have old tree: x,r,s; z,x,y,t,r,s; if we think about s, then we have prefix: x,r; z,x,y,t,r; 
        so it may be hard to find {s,r} because 2 branch may decrease the local support;
        but in new tree, we can re-analyse the data, and may get {r}, then {r,s} with prefix)
        '''
        condPats = findPrefix(basePat, header[basePat])
        subtree, subheader = buildTree(condPats, minSup)

        if subheader!=None:
            print "conditional tree for %s" %(newPrefix)
            subtree.disp()
            mineTree(subtree, subheader, minSup, newPrefix, freqItemList)

'''
FP-Growth is a quick algorithm to find frequent set.
The efficiency of the algorithm comes from ONLY 2 times data scan:
first for global item frequency;
second for tree building;
then all analysis can only base on the FP tree.
Though we need to sort for each record, it is cheap in global.
So building the FP tree is the most hard work in the algorithm - that is a complex structure.

P.S. the Growth comes from the tree building (tree nodes will be created just like tree is growing).
Once the tree built, we could get the frequency set soon.
'''
def FPGrowth(dataset, minSup=1):
    '''
    We will  build a tree and a header table.
    Be careful!
    '''
    tree, header = buildTree(dataset, minSup)
    tree.disp();

    '''
    Q: Why we need header table?
    A: We wanna find the prefix with given item, 
    so we can use the list in header table to find all target in each path of tree, no need to do global search.
    Be careful, we get the tree by frequency (ordered), so we can get the combination (in a prefix) which will meet min support. 
    '''
    #store all frequency here
    freqItemList = []    
    
    #a test flow of the method to find prefix
    for k in header.keys():
        print "prefix of %s: %s" %(k, findPrefix(k,header[k]))

    #recursively mine the FP tree to find frequency set
    mineTree(tree, header, minSup, set([]), freqItemList)
                
    print freqItemList
    return freqItemList
    
if __name__=="__main__":

    filename="FPGrowth-input.txt"
    dataset=loadData(filename)
    print dataset
    newdata = {}
    for record in dataset:
        newdata[record] = newdata.get(record,0) + 1
    print newdata
    
    freqSet = FPGrowth(newdata,3)