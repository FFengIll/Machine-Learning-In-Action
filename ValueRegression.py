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
        for i in datastr:
            data.append(float(i))
        datalist.append(data[0:-1])
        labellist.append(data[-1])

    numFeat = len(datalist[0])

    dataset = array(datalist)
    labelset = array(labellist)
    return dataset, labelset

'''
check the rss error
'''    
def rssError(yArr, yHatArr):
    return ( (yArr-yHatArr)**2 ).sum()
    
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
def standRegress(xArr, yArr):
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
        return None

    xTxI = xTx.I
    ws = xTxI * (xM.T * (weights * yM))
    return target * ws

'''
get the target yHat values using lwlr as regression
'''
def lwlrHat(targetArr, xArr, yArr, k=1.0):
    m = shape(targetArr)[0]
    yHat = zeros(m)
    for i in range(m):
        ret = lwlr(targetArr[i], xArr, yArr, k)
        if(ret == None):
            return None
        yHat[i] = ret
    return yHat
        
'''
ridge regression is a way to do shrinkage while regress
we import the lambda to limit the sum(w)
we import the penalty to reduce the influence of unimportant features (shrinkage)
feathermore, lambda can help to prevent that the matrix can not do inverse
=>
ws= (X.T * X + lambda * I ).I * X.T * y
=>
so we need to min(lambda) - data 1 for test, data 2 for w
'''
def ridgeRegress(xM, yM, lam=0.2):
    xTx = xM.T * xM
    denom = xTx + eye(shape(xM)[1])*lam
    if(linalg.det(denom) == 0.0):
        print "can not do inverse"
        return
    ws = denom.I * (xM.T * yM)
    return ws

'''
ridge regress entry
we have to find the best lambda by test
and for computing, we need to do normalization
=>
2 normalization method:
Min-Max normalization: map into [0-1] (new data may influence the min max)
Z-score normalization: use mean and standard deviation or variance to map Normal Distribution
=>
will return the lambda matrix and related ws matrix
'''
def ridgeRegressWMat(xArr, yArr):
    xM = matrix(xArr).T
    yM = matrix(yArr).T

    #normalize the data
    yMean = mean(yM, 0)
    yM = yM - yMean
    xMean = mean(xM, 0)
    xVar = var(xM, 0)
    xM = (xM -xMean) / xVar
    
    numTestLam = 30
    wMat = zeros( (numTestLam, shape(xM)[1]) )
    lamMat = matrix(range(-10,numTestLam-10)).T
    for i in range(numTestLam):
        #PS: the lambda is number of exp()
        ws = ridgeRegress(xM, yM, exp(i-10))
        wMat[i,:]= ws.T

    return lamMat, wMat

def regular(xM):
    xMean = mean(xM, 0)
    xVar = var(xM, 0)
    xM = (xM -xMean ) / xVar
    return xM

def stageWiseRegress(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)

    #yMat = yMat - yMean
    #xMat = regular(xMat)
    
    m,n = shape(xMat)
    
    res = zeros((numIt,n))   
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    
    #loop times
    for i in range(numIt):
        print ws.T
        lowestErr = inf;
        
        #for each feature, we need to test
        for j in range(n):
            #change:eps or -eps, so the step eps will influence the performance
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssErr = rssError(yMat.A, yTest.A)
            if rssErr < lowestErr:
                lowestErr = rssErr
                wsMax = wsTest
        ws = wsMax.copy()
        res[i,:]=ws.T
        
    return res
    
def preview(xArr,yArr,w):
    fig1 = plot.plot(xArr,yArr,'b*')

    fitfunction = np.poly1d(w)
    newY = fitfunction(xArr)

    fig2 = plot.plot(xArr, newY,'b-')

    plot.show()

def OLSValue(w, x):
    fitfunction = np.poly1d(w)
    yHat = fitfunction(x)    
    return yHat
    
def getMatch(y, yHat):
    res = corrcoef(y, yHat)
    return res


if __name__=="__main__":
    #load data set
    xArr, yArr = file2set("VR-input.txt")
    print xArr
    print yArr   
    
    #prepare x data (just store)
    xTarget = []
    for i in xArr:
        xTarget.append(i[1])
    print xTarget
    
    #OLS
    if(0):
        ws = standRegress(xArr, yArr)
        ws = [ float(v) for v in ws]
        yHat =  [ws[0] + ws[1] * v for v in xTarget]
        
        print "OLS weight:", ws
        print "OLS Regress Value:", yHat
        #exit(0)
    
    #built-in OLS in NumPy
    if(0):
        origin = plot.plot(xTarget,yArr,'b*')
        
        w = OLS(xTarget,yArr)
        yHat = OLSValue(w,xTarget)
        match = getMatch(yArr, yHat)
        
        print "built-in OLS Regress Value:", yHat
        print "built-in OLS Weight:", w
        print "built-in OLS Match:", match
        
        fig2 = plot.plot(xTarget, yHat,'b-')
        plot.show()
        #exit(0)
        
    #LWLR
    if(0):
        origin = plot.plot(xTarget,yArr,'b*')
        print "LWLR origin value:", yArr
        
        #now we do the LWLR test with different parameters
        paraList = [1.0, 0.5, 0.03]
        markType = ['r-', 'g-', 'y-']
        
        for i in range(len(paraList)):
            yHat = lwlrHat(xArr, xArr, yArr, paraList[i])
            if(yHat == None):
                continue
            print "LWLR regress value", yHat
            fig3 = plot.plot(xTarget, yHat, markType[i])
            
        plot.show()
    
    #Ridge Regress
    if(0):
        '''
        We need to regress all features and get the weights to do next analysis.
        Here we only use one as demo.
        The plot will show the weight changing with log(lambda).
        '''
        lamMat ,wMat = ridgeRegressWMat(xArr[:,1], yArr)
        print "Ridge Lambda:",lamMat
        print "Ridge W:", wMat
        plot.plot(lamMat, wMat,'yo',0.03)   
            
        '''
        m,n = shape(xArr)
        lamList = []
        wList = []
        for i in range(n):
            lamMat ,wMat = ridgeRegressWMat(xArr[:,i], yArr)
            print "Ridge Lambda:",lamMat
            print "Ridge W:", wMat
            lamList.append(lamMat)
            wList.append(wMat)
            plot.plot(lamMat, wMat,'yo',0.03)   
        '''
        
        plot.show()    
        
    #Stage Wise Regress
    if(0):
        xArr, yArr = file2set("VR-input2.txt")
        eps = 0.01
        numIt = 500
        wMat = stageWiseRegress(xArr, yArr, eps, numIt)
        #print "Stage Wise W:", wMat
        exit(0)
        plot.plot(lamMat, wMat,'yo',0.03)   
            
        plot.show()        