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
    def __init__(self, args, graph, node_id_feat, node_id_rl, G_features, label_dict, class_count, train_ratio, num_graph, node_ids, labels, pca_features, orig_features, orig_label):
        # (args, G, node_id_feat, node_id_rl, G_features, G_label, class_count, train_ratio, num_graph)
        # CM = ClusteringMachine(args, features, orig_label, class_count, train_ratio, num_graph)
        """s
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.node_seq = {}
        self.subgraph = {}
        self.sg_rl_index = {}
        self.graph = graph
        self.node_ids = node_ids
        self.node_id_rl = node_id_rl
        self.id_feat = node_id_feat
        self.label_dict = label_dict
        self.args = args
        self.features = G_features
        self.num_graph = num_graph
        self.class_count = class_count
        self.ground_truth = labels
        self.train_ratio = train_ratio
        self.pca_feature = pca_features
        self.orig_feature = orig_features
        self.orig_label = orig_label
        self.clusters = [cluster for cluster in range(self.num_graph)]
        # self.index, self.index_gt_map, self.gt = self.data_selt()

    def data_selt(self):
        index = [i for i, value in enumerate(self.ground_truth.reshape(-1, 1)) if value != 0]  #所有非0值所在的索引
        gt_set = self.ground_truth[self.ground_truth != 0]   # 获取非0标签
        node_index_map = {order: idx for order, idx in enumerate(index)}   # 序号：原图id
        return index, node_index_map, gt_set

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            # 图聚类
            self.metis_clustering()
        elif self.args.clustering_method == "generate":
            print("\ngenerator started.\n")
            self.sg_features, self.sg_targets, self.sg_nodes = self.sub_generation(self.args)
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()

    def sub_generation(self, args):
        # 取得非标签像素特征
        features = self.features
        node_ids = self.node_ids # self.index  节点id
        random.seed(args.seed)
        num_selected_nodes = int(np.floor(len(features) / self.num_graph))  # 每个子图的节点数
        # 随机采样
        sg_node_index = []
        subgt_index = []
        sg_feat = []
        gt = self.orig_label.reshape(-1, )
        # gt = torch.tensor(self.orig_label)
        for i in range(self.num_graph):
            temp_index = []
            if (i != self.num_graph - 1):
                selected_nodes = random.sample(node_ids, num_selected_nodes)  # index中随机选取num_selected_nodes个子图节点id
            else:
                selected_nodes = node_ids
            subgt_index.append(torch.LongTensor(gt[selected_nodes]))  # 子图标签对应的是节点id
            sg_node_index.append(selected_nodes)   # 每个子图的节点id
            sg_feat.append(torch.FloatTensor(self.pca_feature[selected_nodes]))   # 子图节点特征
            self.sg_rl_index[i] = {node: i for i, node in enumerate(selected_nodes)}
            node_ids = [node for node in node_ids if node not in selected_nodes]   # 在index中删除第i个子图的节点id
        # 计算类别数
        # Count(subgt_index)
        return sg_feat, subgt_index, sg_node_index   # 得到了子图特征、子图label、子图节点

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """

        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        # 将各个节点属于子图的编号打出
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

