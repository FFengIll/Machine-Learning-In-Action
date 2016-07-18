# MachineLearning
Machine Learning in Action

# Defination
## Classification
## Decision tree
## Regression

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

该算法通过计算香农信息熵来获取最优的特征，从而对数据进行分类——该特征应当是所有特征中，能够得到最优信息增益的一个
# Regression