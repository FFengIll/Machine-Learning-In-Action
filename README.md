# MachineLearning
Machine Learning in Action

* Classification
	* Decision tree
* Regression
* Unsupervised learning
	* Clustering (Unsupervised classification)

# Classification
## KNN (K Nearest Neighbor)
use Euclid distance of data features to do classification
* need to calculate with all samples
* need the normalization to improve accuracy
* simple and valid, though costly

基于欧式距离计算特征的差异进行数据分类
* 需要同全体样本进行计算分析
* 需要对各个特征进行归一化，以确保准确性
* 简单有效，但计算量较大

## ID3 (apply into Decision Tree)

ID3 is applied into Decision Tree, a **greedy algorithm**.

It will calculate the **Shannon Entropy** to get the best features with the best **Information Gain** to classify the data.

ID3算法是用于决策树的一种贪心算法。

该算法通过计算香农信息熵来获取最优的特征，从而对数据进行分类。
所谓最优特征，即是在所有特征中，能够得到最优信息增益的一个

# Regression

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

## Bisection K-Means
