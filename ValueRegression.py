import os,sys
import numpy as np
from numpy import *
from operator import *
import matplotlib
import matplotlib.pyplot as plot

def file2set(filename):
    input=open(filename)

    #store the data(feature) and label(result)
    datalist=[]
    labellist=[]
    for line in input:
        line=line.strip()
        datastr = line.split("\t")
        data = []
        data.append(1.0)
        for i in datastr:
            data.append(float(i))
        datalist.append(data[0:-1])
        labellist.append(data[-1])

    numFeat = len(datalist[0])

    dataset = array(datalist)
    labelset = array(labellist)
    return dataset, labelset

'''
general regression for 2 dimensional data,
aka, OLS (Ordinary Least Square)
=>
sum{ (y-Xw).T (y-Xw) }
=>
let f'(w)=0
then ws = ( X.T X) ^ (-1)  X.T y
=>
but we must have inverse of X
'''
def standRegres(xArr, yArr):
    X = mat(xArr)
    Y = mat(yArr).T
    xTx = X.T * X
    #be careful to use 0, 0.0 may be better
    if linalg.det(xTx) == 0.0:
        print "Det = 0, matrix can not do inverse"
        return
    ws = xTx.I * X.T * Y
    return ws

'''
here use method of Ordinary Least Square in NumPy only
'''
def OLS(xArr, yArr):
    w = np.polyfit(xArr,yArr,1) 
    fitfunction = np.poly1d(w)
    return w

'''
LWLR(Locally Weighted Linear Regression)
Kernel method: Guass Kernel:
w(i,i)= exp( abs( x.(i) - x ) / (-2 * k^2 ) )
the weight will exponentially decaying while distance increasing
(be careful of the negative int, -2)
And the ws changed:
ws = ( X.T * W * X) ^ (-1)  X.T * W * y
'''
def lwlr(target, xArr, yArr, k=1.0):
    xM = mat(xArr)
    yM = mat(yArr).T
    m = shape(xM)[0]
    #init weights as 1
    weights = mat(eye(m))
    
    for i in range(m):
        #get all distance 
        diffM = target - xM[i,:]
        weights[i,i] = exp(diffM * diffM.T / (-2 * k**2 ))
    
    xTx = xM.T * (weights * xM)
    if(linalg.det(xTx)==0.0):
        print "cannot do inverse"
        return

    xTxI = xTx.I
    ws = xTxI * (xM.T * (weights * yM))
    return target * ws

'''
get the yHat values by lwlr
'''
def lwlrHat(targetArr, xArr, yArr, k=1.0):
    m = shape(targetArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(targetArr[i], xArr, yArr, k)
    return yHat
        

def preview(xArr,yArr,w):
    fig1 = plot.plot(xArr,yArr,'b*')

    fitfunction = np.poly1d(w)
    newY = fitfunction(xArr)

    fig2 = plot.plot(xArr, newY,'b-')

    plot.show()

def getMatch(x, y, w):
    fitfunction = np.poly1d(w)
    yMatch = fitfunction(x)
    res = corrcoef(y, yMatch)
    return res


if __name__=="__main__":
    #read data
    xArr, yArr = file2set("VR-input.txt")
    newX = []
    for i in xArr:
        newX.append(i[1])
    print xArr
    print yArr
    print newX
    fig1 = plot.plot(newX,yArr,'b*')

    #OLS
    ws = standRegres(xArr, yArr)
    print ws

    #OLS in NumPy
    w = OLS(newX,yArr)
    match = getMatch(newX, yArr, w)
    print w
    print match
    fitfunction = np.poly1d(w)
    newY = fitfunction(newX)
    fig2 = plot.plot(newX, newY,'b-')

    #LWLR
    yHat = lwlrHat(xArr, xArr, yArr,1.0)
    fig3 = plot.plot(newX, yHat,'r-')
    yHat = lwlrHat(xArr, xArr, yArr)
    fig4 = plot.plot(newX, yHat,'g*',0.5)
    yHat = lwlrHat(xArr, xArr, yArr)
    fig5 = plot.plot(newX, yHat,'yo',0.03)   

    plot.show()