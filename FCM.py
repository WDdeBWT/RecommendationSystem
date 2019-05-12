import pandas as pd
import numpy as np
import random
import operator
import math


class FCM:

    def __init__(self, item_num, item_dim, clusters, max_iter, m = 2.00):
        self.item_num = item_num
        self.item_dim = item_dim
        self.clusters = clusters
        self.max_iter = max_iter
        self.m = m


# 返回的矩阵是每个点属于不同的集群的概率（隶属度）
# 随机生成
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(self.item_num):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat


def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    # cluster_mem_val 's shape is (k, n)
    cluster_centers = list()
    for j in range(k):  # 选定一个类别
        x = cluster_mem_val[j]
        xraised = [e ** m for e in x]  # m is a Fuzzy parameter
        denominator = sum(xraised)  # 求和

        # 将对应点向量乘上它在该类别上的概率 得到概率向量
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)

        # 将所有点在对应特征下的概率累积求和
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        # the shape of center is (num_attr,) , which is the number of the features.
        # 本质上做的是加权平均，关于不同的特征，在不同的点上做加权。权重即为初始概率
        cluster_centers.append(center)
    return cluster_centers


def updateMembershipValue(membership_mat, cluster_centers):
    # m is the Fuzzy parameter.
    p = float(2 / (m - 1))
    # cluster_centers : (k, num_attr)
    for i in range(n):
        x = list(df.iloc[i])
        # 算和两个中心的距离，这里采用的是二范数
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        # 更新概率矩阵
        for j in range(k):
            den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        # cluster_centers: (k, num_attr). k means k clusters. And num_attr means num_attr features.
        # 更新 membership_mat 矩阵
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    # print(membership_mat)
    return cluster_labels, cluster_centers


labels, centers = fuzzyCMeansClustering()
print(labels)
print(len(labels))
# a, p, r = accuracy(labels, class_labels)
#
# print("Accuracy = " + str(a))
# print("Precision = " + str(p))
# print("Recall = " + str(r))