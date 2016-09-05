# MachineLearning
Machine Learning in Action

* Classification
	* Decision tree
* Regression
* Unsupervised learning
	* Clustering (Unsupervised classification)
	* Association analysis (Association rule learning)
	* FP-Growth (An Efficient Frequency Finder in Association rule learning)

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
(Now the FP-Growth implement is not good, because I meet a trouble when some of the support is equal in the item sort)