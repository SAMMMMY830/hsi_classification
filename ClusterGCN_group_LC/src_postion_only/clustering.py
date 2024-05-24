import metis
# import pymetis
import networkx as nx
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
from test import _split_with_min_per_class, _split_with_min_per_class2, _split_with_min_per_class3
from utils import *
import torch.nn.functional as F

class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target, class_count, train_ratio, num_graph):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.node_seq = {}
        self.subgraph = {}
        self.node_index_rl = {}
        self.args = args
        self.graph = graph
        self.nodes = np.array(list(range(features.shape[0])))
        self.features = features
        self.num_graph = num_graph
        self.class_count = class_count
        self.ground_truth = target
        self.train_ratio = train_ratio
        self.layer = np.zeros(self.num_graph)
        self.best_layer = np.zeros(self.num_graph)
        self.clusters = [cluster for cluster in range(self.num_graph)]
        # self.index, self.index_gt_map, self.gt = self.data_selt()

    """
    生成子图
    """
    # def data_selt(self):
    #     index = [i for i, value in enumerate(self.ground_truth.reshape(-1, 1)) if value != 0]  #所有非0值所在的索引
    #     gt_set = self.ground_truth[self.ground_truth != 0]   # 获取非0标签
    #     index_gt_map = {order: idx for order, idx in enumerate(index)}   # 序号：原图id
    #     return index, index_gt_map, gt_set

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        print("\ngenerator started.\n")
        self.sg_features, self.sg_gt, self.sg_nodes = self.sub_generation(self.args,self.num_graph)

    def sub_generation(self, args, num_graph):
        # 取得非标签像素特征
        features = self.features
        # index = self.index  # self.index  节点id
        random.seed(args.seed)
        num_selected_nodes = int(np.floor(features.shape[0] / num_graph))  # 每个子图的节点数
        # 随机采样
        sg_node = []
        sg_gt = []
        sg_feat = []
        index = list(range(self.ground_truth.shape[0]))
        gt = self.ground_truth.reshape(-1, )
        for i in range(num_graph):
            temp_index = []
            if (i != self.num_graph - 1):
                selected_nodes = random.sample(index, num_selected_nodes)  # index中随机选取num_selected_nodes个子图节点id
            else:
                selected_nodes = index
            sg_gt.append(torch.LongTensor(gt[selected_nodes]))  # 子图标签对应的是节点id
            sg_node.append(selected_nodes)   # 每个子图的节点id
            sg_feat.append(torch.FloatTensor(self.features[selected_nodes]))   # 子图节点特征
            self.node_index_rl[i] = {node: i for i, node in enumerate(selected_nodes)}   # 对子图节点创建字典索引，序号：id
            index = [node for node in index if node not in selected_nodes]   # 在index中删除第i个子图的节点id
        # 计算类别数
        # Count(subgt_index)
        return sg_feat, sg_gt, sg_node   # 得到了子图特征、子图label、子图节点
    """
    生成子图
    """