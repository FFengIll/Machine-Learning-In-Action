# Machine Learning in Action

# Abstract
A note from **Machine Learning in Action**.

Context may not follow the list, and English go ahead Chinese.

本文参考文献：《机器学习实战》。

正文可能并未按照目录书写。全文各段落英文在前，中文在后。

# Tips
机器学习的功用：预测与分类。

# List
- [Machine Learning in Action](#machine-learning-in-action)
- [Abstract](#abstract)
- [Tips](#tips)
- [List](#list)
- [Classification](#classification)
	- [KNN (K Nearest Neighbor)](#knn-k-nearest-neighbor)
	- [ID3 (apply into Decision Tree)](#id3-apply-into-decision-tree)
	- [Naive Bayes](#naive-bayes)
- [SVM (Support Vector Machine)](#svm-support-vector-machine)
- [AdaBoost](#adaboost)
- [Logistic Regressive](#logistic-regressive)
- [Linear Regression](#linear-regression)
	- [Brief](#brief)
	- [OLS (Ordinary Least Squares)](#ols-ordinary-least-squares)
	- [LWLR (Locally Weighted Linear Regression)](#lwlr-locally-weighted-linear-regression)
	- [Ridge Regression](#ridge-regression)
	- [Lasso Regression](#lasso-regression)
	- [Stage Wise Regression](#stage-wise-regression)
	- [Summary](#summary)
- [Tree Regression](#tree-regression)
	- [CART](#cart)
- [Unsupervised learning](#unsupervised-learning)
	- [K-Means](#k-means)
	- [Bisecting K-Means](#bisecting-k-means)
	- [Apriori Algorithm](#apriori-algorithm)
	- [FP-Growth](#fp-growth)

# Classification
## KNN (K Nearest Neighbor)
use Euclid distance of data features to do classification
* need to calculate with all samples
* need the normalization to improve accuracy
* simple and valid, though costly

基于欧式距离计算特征的差异进行数据分类
* 需要同全体样本进行计算分析
* 需要对各个特征进行归一化，以确保准确性
* 简单有效，但各个步骤的计算量都较大

## ID3 (apply into Decision Tree)

ID3 is applied into Decision Tree, a **greedy algorithm**.

It will calculate the **Shannon Entropy** to get the best features with the best **Information Gain** to classify the data.

ID3算法是用于决策树的一种贪心算法。

该算法通过计算香农信息熵来获取最优的特征，从而对数据进行分类。
所谓最优特征，即是在所有特征中，能够得到最优信息增益的一个。

PS：香农熵越大，意味着信息越丰富；
香农熵越小，则信息越明确。
最优信息增益，即分类使得整体熵下降，信息被明确划分。

## Naive Bayes
基于统计事实和概率的分类算法。核心是贝叶斯定理，具有较强的模型可解释性。

# SVM (Support Vector Machine)
一个（个人认为）“玄之又玄”的算法，核心是kernel算子的选用（使得数据低维非线性而高维线性可分割）。

可用于模式识别，回归，分类。

# AdaBoost
通过训练多个弱分类器，从而整合，形成具有强分类效果的迭代算法。

弱分类器是指分类效果相对单一，使用部分测试集的分类模型，一般通过（加权）投票表决的方式整合。

# Logistic Regressive
主要用于分类

> https://www.cnblogs.com/weiququ/p/8085964.html

# Linear Regression
主要用于预测

## Brief
I will not do too much explanation on linear regression algorithm,
but explain the core idea in all of them here.

Whatever the algorithm we use, we just wish to get the trend of the data.
So we should take care of 2 points: 
* overfit will give too much info and cause model complex;
* underfit may lose the accuracy (simple model), then be hard to use.

线性回归的核心理念是：通过已有数据，获取数据的趋势情况，应用于预测、验证、评估等场景中。
在线性回归中，需要特别注意两点：
* 过拟合，会导致信息量过于丰富，进而使得模型过于复杂（如因素或特征过多）。
* 欠拟合，其模型会较为简单，但相应的也就可能使得精读丢失严重，而无法使用。

## OLS (Ordinary Least Squares)
所谓最小二乘法

## LWLR (Locally Weighted Linear Regression)

## Ridge Regression

## Lasso Regression
Lasso（Least Absolute Shrinkage and Selection Operator）, somehow like the Ridge but use absolute value.

## Stage Wise Regression

## Summary


# Tree Regression
## CART
Classification And Regression Trees will give regression by slice data into a tree structure.
The core idea is that slice data (**especially the Non-Linear data**) into more simple subset,
and then use the linear regression for each subset.

CART can support the complex data (global non-linear but local linear) well and each sub-mission can be small, easy and local.
And here we give it:
* find the best value to do the split
	* traverse all features
	* for each features, get the feasible value as a split set
	* for each split value in the split set, try to split and compute the error of subset
	* if the sum of the 2 subset error is less than the origin one, update the status
* then we have the best split feature and value, so split the dataset into 2 subsets with it
* do the find and split the new set recursively. The terminate is any of the conditions bellow: (avoid overfitting)
	* error decline is not obvious, aka under **tolerance error**.
	* the subset is under **tolerance scale** (aka number of sample).
* then we have the CART of the dataset

CART work well, but it often cause overfitting when the tree nodes are to many, so we need pruning.
Here comes 2 ways to do it: prepruning and postpruning.

The conditions above are prepruning:
* if error decline little, the split can help little;
* if samples are not enough, the regression will not work well.

And we can use testing set and training set to complete cross validation as postpruning:

分类回归树（CART）采用了数据集划分的思想，来对数据集进行分类以用于回归，即：
将数据集划分为多个简易的，可线性回归的子集，进而进行局部的线性回归。
这样非常有益于处理复杂的非线性问题（全局非线性，无法进行全局拟合，但局部线性，可以局部线性拟合）。

CART的基本算法如下：
* 试探最优的划分特征及特征值
	* 遍历全部特征
	* 对每一个特征，获取划分集合，即所有备选特征值
	* 使用每一个特征值进行集合划分，并计算子集的偏差
	* 如果子集的偏差之和低于原始集合，则更新状态
* 而后即可获得最优的划分特征及特征值，并进行数据集划分
* 通过递归的对每一个新的数据集进行划分，即可完成CART的构建. 其递归结束条件为以下任一（防止过拟合）：
	* 误差的下降不明显，即减少量低于**容忍误差**。
	* 子集中的样本数目低于**容忍样本数目**。
* 最后，完成CART构建，即对原始数据的划分。

# Unsupervised learning

## K-Means
As a Clustering Algorithm, K-Means try to use the centroids to stand for the clusters - 
each cluster will belong to one centroid.

So the number of centroids just means the number of clusters (that is the K).

The main k-means algorithm is shown bellow:
* randomly choose/give k centroids (must in the valid range)
* calculate all distances between data and centroids, and choose the best one centroid for each node
* re-calculate the centroids according to the clusters
* go on loop until the clusters become stable (no change) 

PS: we can choose the distances calculation method, 
but it will significantly influence the algorithm (i.e. it is the core).

K-均值聚类算法属于聚类算法，其核心思想是使用质心来代表整个簇的数据，即每个簇中的数据都会归属于一个质心。

所以，质心的数目即为聚类中簇集的数目（即K）。

K-均值的算法简述如下：
* 给定K个质心（随机或指定，但质心应当在数据有效值域内）
* 计算每个数据点到各个质心的距离，并选择最优的质心作为数据点的归属的簇
* 由于数据被重新分类，实际质心也发生了变化，所以需要重新计算质心（即均值）
* 循环计算（即迭代：计算质心-分配-重新计算），直到聚类中的簇不再发生变化，所有分类达到稳定状态即可

需要特别注意：允许采用任意的距离度量方法，但是度量方法会显著影响算法性能（大量的计算调用）

## Bisecting K-Means
Same work flow as K-Means, but has another basic idea -
we will try to do Bisecting in each loop but directly output k centroids.

In each loop, we will choose the best Bisecting which will decrease the global SSE,
and accept this Bisecting in the loop until we get K clusters.

Futhermore, the SSE is the sum of squared error, which reflects the quality and accuracy (we can compute it with different methods).

二分K均值算法是对K均值的一种改进：并非直接求解K个质心，而是每次选取一个簇，
并对簇进行二分（即进行k=2的k均值聚类）；通过遍历当前所有簇，找到能够使得SSE最小化的二分，
从而更新整个聚类；直到找到K个簇为止。

通过逐步二分聚类，可以比较有效的克服K-均值算法中收敛于局部最小值的问题（原因有随机质心，逐步更新，质心迁移等）。

备注：上文中的SSE，即sum of squared error，平方误差和，其计算方法可以选择，一般与K-均值中的方法一致。

## Apriori Algorithm
(I think my implement is not good enough, so pls contact with me if you have something better)

Apriori has a target to get the association from data, so it is association rule learning.

We always wanna know the **_implicit_** relationship between things, especially in these situations: and so on.

In Apriori, we generally follow the steps:
* find the set of high frequency (we call the frequency as support), meanwhile, we will calculate the support value.
	* support is easy to calculate: occurrences of the set / total data items, aka the frequency of the set.
	* if the the data item include the set, we say the set occurrents.
	* (optimization) if the set is non-frequent, its superset must be non-frequent too.
	* (optimization) save the support, it will be used later.
* find and extract the rules from frequency set, and believe the rules by confidence value
	* confidence can be calculated using support value before.
	* for a layer, if a consequence does not meet, all superset as the consequence can not meet either.
	* e.g. for N = X + Y, if X->Y does not meet, then X - delta -> Y + delta does not meet, where delta belong to X

Apriori算法属于关联分析算法，亦即关联规则学习算法。

之所以使用关联分析，是因为大量的数据之间存在着潜在的联系，而这一潜在的数据关联即反映了现实中事物的隐形关联。

Apriori算法一般包含两个步骤：
* 分析频繁项集。 同时，还可以计算支持度（support）
	* 支持度的计算即：集合出现频数 / 总数据项， 亦即集合的出现频率。
	* 如果集合或其子集为目标集合，即该集合计数加一。
	* （优化）如果一个集合是非频繁的，其超集必然非频繁。
	* （优化）可以将支持度保存，并在后续使用（置信度计算依赖于支持度）。
* 分析并抽取规则。规则需要根据频繁项集去分析和抽取，并通过置信度（confidence）决定是否信赖该规则。
	* 置信度可以根据支持度计算，如：C(AB->C) = C(ABC) / C(AB)。
	* 对于每一层频繁项（同等大小的集合位于一层），如果某一集合作为结论的规则不可置信，则其超集作为结论也不可置信。
	* 如：N中，X+Y = N，如果X->Y不可置信, 则 X - delta -> Y + delta 亦不可置信，其中delta属于X。 

## FP-Growth
~~(Now the FP-Growth implement is not good, because I meet a trouble when some of the support is equal in the item sort)~~

(I found I have misunderstand the FP-Tree's application - each time we do a new analysis, we should re-build a FP-Tree and scan it in recursion.
That is saying if some item in the FP-Tree may have same support, then the order is undefined, 
but we can rebuild the tree later to find which one, same upport in last tree, is more frequent)

FP-Growth is an efficient algorithm to find out the frequent item in the dataset.
It is different with Apriori, because FP-Growth should and only should scan the dataset twice!
That means we can considerably reduce the dependency of dataset and time cost (especially in characters/string)

FP-Growth bases on the FP-Tree, aka Frequency Pattern Tree, and extract the frequent item from the FP-Tree according to the conditional pattern.
FP-Tree is built and update node by node, that is why we call it Growth. 

Then I will introduce FP-Growth in 2 aspects:
* FP-Tree building. It is the core of the algorithm.
	* Analyse the dataset, get the support (frequency) of each item in each record (the first data scan) 
	* By the given min support, filter the items
	* Scan all record and build the FP-Tree (the second data scan).
		* Fileter and sort the items in record by item support (from large to little) - to avoid divergency (e.g. xyz, zyx may cause trouble without sort)
		* From the empty tree root node, while scan the record, if the item node exist, update the local support count attribute, else add a new node.  
		Then we will have all ordered path in the tree.
		* For some reason, we may build a header table - all item in the header and link to a list which is a link list of the item in the tree.  
		It will help us to find the items quickly.
	* Now we have the tree and the header table  
* FP-Growth
	* If an item is frequent, its prefix in one FP-Tree may be frequent too.  
	* So we scan from all item (from little to large) - each item is a base pattern and we get its all prefixs as conditional pattern in the tree.
		* Each time we scan the item, we append it to current set (empty set at init) as a new frequency set, and save it - so we call the process as **_Tree Mine_**.
		* Then we use conditional pattern to build a new tree - FP conditional tree (we have min support!)
		* If the tree is not empty, go on scan and mine the frequency set. (**_It is a recursion_**) 


FP-Growth算法是用于发掘频繁项集的高效算法。

相比于Apriori算法，FP仅需要对数据集进行两次扫描，因此其对数据的依赖会更低，同时效率会高出很多（一般会出现数量级的超出），特别的，在字符处理上，FP的表现更佳。

FP-Growth算法的核心是树状结构的，即频繁模式树（FP-Tree），在算法中，频繁项的抽取是直接依赖于模式树中的条件基的（后续说明）。
而FP-Tree的构建和更新是逐个节点完成的，如同树的生长，故而称之为FP-Growth。

FP-Growth算法主要包含了两个方面：
* FP-Tree的构建。
	* 分析数据集，获取每条记录中的每项的支持度（以数据集为字符串记录为例）（第一次扫描）
	* 根据最小支持度过滤元素
	* 扫描数据，并构建FP-Tree（第二次扫描）
		* 对每条记录进行过滤和排序（支持度从大到小），进而防止顺序不同而出现分歧；
		* 以空节点为根（具体实现可不同），进行树的生长。扫描记录时，若节点存在，则更新局部支持度，继续探索子节点；若节点不存在，则新建子节点；
		* 在树的生长期间，需要创建头结点表——将所有有效元素存储与头表中，每一个元素链接到一个链表之中，链表为FP-Tree中的全部的这一个元素项（有助于加速搜索）；
		* 当全部记录扫描完毕，即可得到树及头表。
* FP-Growth。（递归进行）（看看代码，辅助理解）
	* 如果某一元素是频繁的，其在树中的前缀则也可能是频繁的；
	* 我们可以循环扫描所有频繁单项（即头表中的元素）（支持度由小到大，因为大支持度一般更靠近树根）（这是一个挖掘树中信息的过程）。  
	每一个元素视作模式基，其在树中的全部前缀即为条件模式；
		* 每当我们扫描一个元素，则将其加入当前集合中，该集合即一个新的频繁项集；
		* 而后，获取该元素条件模式，并意条件模式作为局部数据集，构建FP-Conditional Tree；
		* 如果新的条件树存在，则可以递归的对频繁项集进行挖掘；