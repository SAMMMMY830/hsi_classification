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
    def __init__(self, args, features, target, class_count, train_ratio, num_graph):
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
        # self.graph = graph
        self.features = features
        # self.target = target
        self.num_graph = num_graph
        # self._set_sizes()
        self.class_count = class_count
        self.ground_truth = target
        self.train_ratio = train_ratio
        self.layer = np.zeros(self.num_graph)
        self.best_layer = np.zeros(self.num_graph)
        self.clusters = [cluster for cluster in range(self.num_graph)]
        self.index, self.index_gt_map, self.gt = self.data_selt()

    # def _set_sizes(self):
    #     """
    #     Setting the feature and class count.
    #     """
    #     self.feature_count = self.features.shape[1]
    #     self.class_count = np.max(self.target)+1
    def data_selt(self):
        index = [i for i, value in enumerate(self.ground_truth.reshape(-1, 1)) if value != 0]  #所有非0值所在的索引
        gt_set = self.ground_truth[self.ground_truth != 0]   # 获取非0标签
        index_gt_map = {order: idx for order, idx in enumerate(index)}   # 序号：原图id
        return index, index_gt_map, gt_set

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            # 图聚类
            self.metis_clustering()
        # elif self.args.clustering_method == "louvain":
        #     print("\nLouvain graph clustering started.\n")
        #     self.louvain_clustering()
        elif self.args.clustering_method == "generate":
            print("\ngenerator started.\n")
            self.sg_features, self.sg_targets, self.sg_nodes = self.sub_generation(self.args)
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()

    def sub_generation(self, args):
        # 取得非标签像素特征
        features = self.features[self.index]
        index = self.index  # self.index  节点id
        random.seed(args.seed)
        num_selected_nodes = int(np.floor(features.shape[0] / self.num_graph))  # 每个子图的节点数
        # 随机采样
        node_index = []
        subgt_index = []
        sub_feat = []
        gt = self.ground_truth.reshape(-1, )
        for i in range(self.num_graph):
            temp_index = []
            if (i != self.num_graph - 1):
                selected_nodes = random.sample(index, num_selected_nodes)  # index中随机选取num_selected_nodes个子图节点id
            else:
                selected_nodes = index
            subgt_index.append(torch.LongTensor(gt[selected_nodes]))  # 子图标签对应的是节点id
            node_index.append(selected_nodes)   # 每个子图的节点id
            sub_feat.append(torch.FloatTensor(self.features[selected_nodes]))   # 子图节点特征
            self.node_index_rl[i] = {node: i for i, node in enumerate(selected_nodes)}   # 对子图节点创建字典索引，序号：id
            index = [node for node in index if node not in selected_nodes]   # 在index中删除第i个子图的节点id
        # 计算类别数
        # Count(subgt_index)
        return sub_feat, subgt_index, node_index   # 得到了子图特征、子图label、子图节点

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
        # 对节点进行聚类，完成切割子图操作
        # adjacency_list = {n: list(self.graph.neighbors(n)) for n in self.graph.nodes()}
        # (st, parts) = pymetis.part_graph(self.args.cluster_number, adjacency_list)

        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        # 将各个节点属于子图的编号打出
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def pre_train(self,node, edge, features, labels,adj):
        # 取节点和边的数量
        node_sum = adj.shape[0]
        edge_sum = adj.sum() / 2
        row_sum = (adj.sum(1) + 1)
        # 度的归一化操作
        norm_a_inf = row_sum / (2 * edge_sum + node_sum)

        # 对adj进行增强随机游走并将结果转换为稀疏张量，随机游走可以增强节点之间的连接性
        adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))
        # 使用F1范数对其进行归一化处理，使得不同的节点的特征相量具有相同的尺度比，在0-1范围内。
        features = F.normalize(features, p=1)
        #
        feature_list = []
        feature_list.append(features)
        # 对其进行特征平滑，次数为K次
        for i in range(1, self.args.k1):
            feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
        # 转换形状，后续需要与特征相乘
        norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
        # 对特征进行归一化，平滑不同节点的度数对聚合的影响
        norm_fea_inf = torch.mm(norm_a_inf, features)
        # 定义一个长度为node的跳数
        hops = torch.Tensor([0] * (adj.shape[0]))
        # mask_before用来记录当前节点是否被更新过，初始为FALSE
        mask_before = torch.Tensor([False] * (adj.shape[0])).bool()

        for i in range(self.args.k1):
            # 计算每个节点与当前轮次的中心节点的欧几里得距离，其中中心节点是通过第i次特征平滑得到的节点特征向量
            dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
            # 这里是将没有找到迭代轮次的设为True，找到的设为False。
            mask = (dist < self.args.epsilon1).masked_fill_(mask_before, False)
            mask_before.masked_fill_(mask, True)
            # 将当前距离小于距离特定值的用i来代替，这里的i就是当前节点的迭代次数，此时的mask为True的点都为继续迭代的点，即没有进入最小距离阈值。
            hops.masked_fill_(mask, i)
        # 将所有 mask_before 值为 True 的节点对应的 mask_final 值更新为 False，此时为True的节点，迭代次数为 k1-1，表示这些节点距离中心节点的距离超过了所有轮次的阈值。
        mask_final = torch.Tensor([True] * (adj.shape[0])).bool()
        mask_final.masked_fill_(mask_before, False)
        hops.masked_fill_(mask_final, self.args.k1 - 1)
        print("Local Smoothing Iteration calculation is done.")

        # 取hops跳的特征平滑。
        input_feature = aver(adj, hops, feature_list)
        print("Local Smoothing is done.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_class = (labels.max() + 1).item()
        input_feature = input_feature.to(device)

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        # 定义子图参数
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        self.sg_edegsfea={}
        self.sg_adj={}
        self.sg_graph={}
        self.degree = {}
        self.degree_max = {}
        self.edge = 0
        self.index_graph={}
        self.matrix={}
        self.ori_index={}
        # G = self.graph
        # G_list = nx.to_dict_of_lists(G)
        # adj = nx.adjacency_matrix(nx.from_dict_of_lists(G_list))
        # 对每个子图执行下述操作
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            self.sg_adj[cluster] = np.zeros((len(self.sg_nodes[cluster]), len(self.sg_nodes[cluster])))
            # 对子图中的节点做映射
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.ori_index[cluster] = mapper
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            # 创建带有权重的adj
            for edge in self.sg_edges[cluster]:
                u, v = edge
                self.sg_adj[cluster][u][v] = 1

            # self.sg_adj[cluster] = self.cre_adj(subgraph, mapper ,cluster)
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster], :]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster], :]
            # 对当前子图的训练集和测试集做划分
            # self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.args.test_ratio)
            # self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = _split_with_min_per_class(X=self.sg_features[cluster], y=self.sg_targets[cluster], test_size=self.args.test_ratio)
            # 输出转换后的矩阵
            self.edge = self.edge + len(self.sg_edges[cluster])
            self.degree[cluster] = self.sg_adj[cluster].sum(0)
            self.degree_max[cluster]=self.degree[cluster].max()

            clustering_coefficients = nx.clustering(subgraph).values()
            clustering_coefficients_list = list(clustering_coefficients)
            # 绘制图结构
            pos = nx.spring_layout(subgraph)  # 选择节点的布局算法
            # 将 clustering_coefficients 转换成列表，并按照节点顺序排列
            # nodes = list(subgraph.nodes())
            # clustering_coefficients_list = list(clustering_coefficients[node] for node in nodes)
            # clustering_coefficients_list = [clustering_coefficients[node] for node in nodes]
            # 绘制图结构，并使用节点的聚类系数进行着色
            # nx.draw_networkx_nodes(subgraph, pos, node_color=clustering_coefficients_list, cmap='viridis',
            #                        node_size=100)
            # plt.axis('off')# 禁用坐标轴
            # plt.savefig('../image/'+str(cluster), bbox_inches='tight')
            # plt.show()
            self.index_graph[cluster] = nx.average_clustering(subgraph)
            # print("子图{}的聚类系数为：{}".format(cluster, self.index_graph[cluster]))
            # print("当前子图度{}}".format(self.degree))
            # print("当前子图中的训练集数量{},当前子图边的数量{}, 当前类别数量{}".format(len(self.sg_train_nodes[cluster]),len(self.sg_edges[cluster]), class_num))


    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()

            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])
            self.degree[cluster] = torch.LongTensor(self.degree[cluster])
            self.sg_adj[cluster] = torch.FloatTensor(self.sg_adj[cluster])
            # self.index_graph[cluster] = torch.LongTensor(list(self.index_graph[cluster]))
            # self.pre_train(self.sg_nodes[cluster], self.sg_edges[cluster],
            #                self.sg_features[cluster], self.sg_targets[cluster],
            #                self.sg_adj[cluster])



    def cre_adj(self, subgraph, mapper, cluster):
        # 获取子图中所有边的起始节点和目标节点
        edges = np.array(subgraph.edges())

        # 获取边权重
        edge_weight = np.array([subgraph[edge[0]][edge[1]]["e_fet"] for edge in edges])

        # 构建映射后的带权重的邻接矩阵
        adj_matrix = np.zeros((len(self.sg_nodes[cluster]), len(self.sg_nodes[cluster])))

        # 获取边的起始节点和目标节点索引
        node_indices = np.array([[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()])

        # 在邻接矩阵中赋值边的权重
        adj_matrix[node_indices[:, 0], node_indices[:, 1]] = edge_weight
        # adj_matrix[node_indices[:, 1], node_indices[:, 0]] = edge_weight  # 无向图需设置对称元素
        return adj_matrix

