import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from modules import norm_layer
from src import graphlearn
from src.utils import sptial_neighbor_matrix, compute_dist
from graph_learn import learn_graph
from graphlearn import GraphLearner
from GCNlayer import GCNLayers,MLP
from para_model import Para_model
import Outils.multiple_embedding as mutiple
from position import PositionEncoding as PE, PositionEncoding, PClassifier
from position import LSTM_pooling as LSTM

class DeeperGCN(torch.nn.Module):

    def __init__(self, args, adj, node_feat_dim, hid_dim, PE, nodes,out_dim,norm='batch', beta=1.0, clusters = 0, degree_max=0):

        # self.model = DeeperGCN(
        #     args=self.args,
        #     adj=self.adj,
        #     node_feat_dim=self.features.shape[1],
        #     hid_dim=self.args.hid_dim,
        #     out_dim=self.class_count,
        #     clusters=self.clusters,
        #     degree_max=self.sg_nodes,
        #     model2=self.PE
        # )
        super(DeeperGCN, self).__init__()
        # para_init

        self.args = args
        self.nodes = nodes
        # self.num_layers = num_layers
        self.dropout = self.args.dropout
        self.norm = norm
        self.pool = args.pool
        self.hid_dim = hid_dim
        self.node_feat_dim = node_feat_dim
        self.module_num = 1
        self.class_num = out_dim
        self.clusters = clusters
        # model_init

        self.graph_skip_conn = Para_model()
        # model_para
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.d_norms = nn.ModuleList()
        self.embedding=nn.ModuleList()
        self.linear = nn.ModuleList()
        self.grouping = nn.ModuleList()
        self.mlp = nn.ModuleList()
        # self.graph_learner = GraphLearner(node_feat_dim, hid_dim, device=self.args.cuda, epsilon=self.args.epsilon)
        self.graph_learner = nn.ModuleList()
        self.para_model = []
        # 聚类系数映射
        self.idx_graph = nn.ModuleList()
        self.bilstm = nn.ModuleList()
        self.bns= nn.ModuleList()
        self.in_emb_dim = self.module_num * self.hid_dim
        bn = torch.nn.BatchNorm1d(self.hid_dim)
        self.embedding_layer = nn.Embedding(
            num_embeddings= len(self.nodes),
            embedding_dim= self.hid_dim,
            padding_idx=0
        )
        # for cluster in self.clusters:
        #     self.embedding.append(nn.Embedding(len(degree_max[cluster]), hid_dim).to(self.args.cuda))
        #     self.idx_graph.append(nn.Embedding(len(degree_max[cluster]), hid_dim))
        #     self.graph_learner.append(GraphLearner(node_feat_dim, hid_dim, device=self.args.cuda, epsilon= self.args.epsilon))
        self.PE = PE
        self.conv1 = GCNLayers(hid_dim*2, hid_dim)
        self.conv2 = GCNLayers(hid_dim, hid_dim)
        for i in range(self.args.num_layers):
            self.bns.append(bn)
        self.linear = nn.Linear(node_feat_dim, self.class_num)
        self.bilstm = LSTM(args)
        if args.grouping_layer == 'mul_linear':
            self.grouping.append(mutiple.MLGrouping(args, self.in_emb_dim))
        else:
            self.grouping.append(mutiple.QGrouping(args, self.in_emb_dim))
        self.mlp = MLP(self.hid_dim*2, self.hid_dim)
        self.classifier = PClassifier(args, self.class_num)


    def forward(self, sg_nodes,node_feats, train_nodes, cluster, mode, degeree, layer, init_adj, target):
        init_feat = node_feats
        layer = int(layer)
        # 获取标签类别的数量
        num_classes = self.class_num
        device = target.device
        # one_hot_matrix = torch.eye(num_classes).to(device)[target.flatten()]
        dist = 0
        if mode == 'train':
            # loss, layer = self.train_epoch1(node_feats, train_nodes, init_adj, layer, target, init_feat, cluster)
            totol_loss= self.train_epoch2(sg_nodes, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster)
            return totol_loss
        else:
            output = self.test_epoch2(sg_nodes, node_feats, init_adj, layer, cluster)
            return output

    def learn_adj(self, init_adj, feature, cluster, training = True):
        cur_raw_adj, cur_adj =  learn_graph(self.args, self.graph_learner[cluster], feature, graph_skip_conn=self.args.graph_skip_conn, init_adj=init_adj)
        cur_raw_adj = F.dropout(cur_raw_adj, self.args.feat_adj_dropout, training=training)
        cur_adj = F.dropout(cur_adj, self.args.feat_adj_dropout, training=training)
        return cur_raw_adj, cur_adj

    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        # mx[np.nonzero(mx)] = 1
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def add_graph_loss(self, out_adj, features, dist):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        # 这里是最小化Ω（X,A），其中self.config['smoothness_ratio'] 为a
        out_adj = self.to_cuda(out_adj, self.args.cuda)
        dist = self.to_cuda(out_adj, self.args.cuda)
        graph_loss += self.args.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        graph_loss += self.args.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def get_degree(self, adj):
        d = adj.sum(0).long()
        return d

    def train_epoch2(self, sg_nodes, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster):
        # self.train_epoch2(sg_nodes, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster, one_hot_matrix)
        # init_adj = self.normalize_adj(init_adj)
        spec_fea = node_feats
        # node_num = node_feats.size(0)

        node_position_embedding = self.PE(sg_nodes)   # 每个子图中节点的位置编码
        sg_nodes_em = torch.tensor(sg_nodes).to(self.args.device)
        node_neibor_embedding = torch.cat((node_position_embedding, self.embedding_layer(sg_nodes_em)), dim = 1)   # 拼接NP
        norm_adj = self.normalize_adj(init_adj)
        node_embedding = node_neibor_embedding
        # get node_embedding
        node_embedding = F.relu(self.conv1(node_embedding, norm_adj))
        node_embedding = self.bns[0](node_embedding)
        for i in range (self.args.num_layers - 1):
            spec_fea = F.relu(self.conv2(spec_fea, norm_adj))
            spec_fea = self.bns[i+1](spec_fea)
        node_position_embedding = torch.unsqueeze(node_position_embedding, dim=0)
        # structure-aware embedding
        structure_embedding = self.bilstm(node_position_embedding)
        # structure_embedding = self.mlp(structure_embedding)

        # 1: 将结构信息加到每个节点特征上
        # pred = torch.add(node_embedding, structure_embedding)
        # 2: 将结构信息拼接到节点特征上
        # structure_embedding = structure_embedding.expand(node_embedding.size())
        # pred = torch.cat((node_embedding, structure_embedding), dim=-1)
        # pred = self.mlp(pred)
        # 3:PClassifier
        pred = self.classifier(node_embedding, structure_embedding, spec_fea)

        output = torch.nn.functional.log_softmax(pred, dim=1)
        loss = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes]-1)
        # loss1 = loss1 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
        # # 循环准备
        # first_adj = cur_adj
        totol_loss = loss
        return totol_loss


        pred = self.classifier(node_embedding, structure_embedding, spec_fea)

        output = torch.nn.functional.log_softmax(pred, dim=1)
        loss = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes]-1)
        # loss1 = loss1 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
        # # 循环准备
        # first_adj = cur_adj
        totol_loss = loss
        return totol_loss


    # def train_epoch3(self, sg_nodes, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster):
    #     # self.train_epoch2(sg_nodes, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster, one_hot_matrix)
    #     # init_adj = self.normalize_adj(init_adj)
    #     spec_fea = node_feats
    #     # node_num = node_feats.size(0)
    #
    #     node_position_embedding = self.PE(sg_nodes)   # 每个子图中节点的位置编码
    #     sg_nodes_em = torch.tensor(sg_nodes).to(self.args.device)
    #     node_neibor_embedding = torch.cat((node_position_embedding, self.embedding_layer(sg_nodes_em)), dim = 1)   # 拼接NP
    #     norm_adj = self.normalize_adj(init_adj)
    #     node_embedding = node_neibor_embedding
    #     # get node_embedding
    #     node_embedding = F.relu(self.conv1(node_embedding, norm_adj))
    #     node_embedding = self.bns[0](node_embedding)
    #     for i in range (self.args.num_layers - 1):
    #         spec_fea = F.relu(self.conv2(spec_fea, norm_adj))
    #         spec_fea = self.bns[i+1](spec_fea)
    #     node_position_embedding = torch.unsqueeze(node_position_embedding, dim=0)
    #     # structure-aware embedding
    #     structure_embedding = self.bilstm(node_position_embedding)
    #     # structure_embedding = self.mlp(structure_embedding)
    #
    #     # 1: 将结构信息加到每个节点特征上
    #     # pred = torch.add(node_embedding, structure_embedding)
    #     # 2: 将结构信息拼接到节点特征上
    #     # structure_embedding = structure_embedding.expand(node_embedding.size())
    #     # pred = torch.cat((node_embedding, structure_embedding), dim=-1)
    #     # pred = self.mlp(pred)
    #     # 3:PClassifier
    #     pred = self.classifier(node_embedding, structure_embedding, spec_fea)
    #
    #     output = torch.nn.functional.log_softmax(pred, dim=1)
    #     loss = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes]-1)
    #     # loss1 = loss1 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
    #     # # 循环准备
    #     # first_adj = cur_adj
    #     totol_loss = loss
    #     return totol_loss
    #
    #
    #     pred = self.classifier(node_embedding, structure_embedding, spec_fea)
    #
    #     output = torch.nn.functional.log_softmax(pred, dim=1)
    #     loss = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes]-1)
    #     # loss1 = loss1 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
    #     # # 循环准备
    #     # first_adj = cur_adj
    #     totol_loss = loss
    #     return totol_loss

    def test_epoch2(self, sg_nodes, node_feats, init_adj, layer, cluster):
        spec_fea = node_feats
        # node_num = node_feats.size(0)

        node_position_embedding = self.PE(sg_nodes)  # 每个子图中节点的位置编码
        sg_nodes_em = torch.tensor(sg_nodes).to(self.args.device)
        node_neibor_embedding = torch.cat((node_position_embedding, self.embedding_layer(sg_nodes_em)), dim=1)  # 拼接NP
        norm_adj = self.normalize_adj(init_adj)
        node_embedding = node_neibor_embedding
        # get node_embedding
        node_embedding = F.relu(self.conv1(node_embedding, norm_adj))
        node_embedding = self.bns[0](node_embedding)
        for i in range(self.args.num_layers - 1):
            spec_fea = F.relu(self.conv2(spec_fea, norm_adj))
            spec_fea = self.bns[i + 1](spec_fea)
        node_position_embedding = torch.unsqueeze(node_position_embedding, dim=0)
        # structure-aware embedding
        structure_embedding = self.bilstm(node_position_embedding)
        # structure_embedding = self.mlp(structure_embedding)

        # 1: 将结构信息加到每个节点特征上
        # pred = torch.add(node_embedding, structure_embedding)
        # 2: 将结构信息拼接到节点特征上
        # structure_embedding = structure_embedding.expand(node_embedding.size())
        # pred = torch.cat((node_embedding, structure_embedding), dim=-1)
        # pred = self.mlp(pred)
        # 3:PClassifier
        pred = self.classifier(node_embedding, structure_embedding, spec_fea)

        output = torch.nn.functional.log_softmax(pred, dim=1)
        return output

    def add_network(self, layer, cur_feat):
        if len(self.gcns) <= layer + 1:
            # self.gcns.append(GCN(self.node_feat_dim, 32, self.node_feat_dim, self.dropout)).to(self.args.cuda)
            self.gcns.append(GCNLayers(self.node_feat_dim, self.node_feat_dim)).to(self.args.cuda)
            self.norms.append(norm_layer(self.norm, self.hid_dim)).to(self.args.cuda)
            self.linear.append(nn.Linear(cur_feat.shape[1], cur_feat.shape[1])).to(self.args.cuda)
            self.d_norms.append(norm_layer(self.norm, self.hid_dim)).to(self.args.cuda)
            self.mlp.append(MLP(self.args.embedding_dim[0], self.hid_dim)).to(self.args.cuda)
            if self.args.grouping_layer == 'mul_linear':
                self.grouping.append(mutiple.MLGrouping(self.args, self.in_emb_dim)).to(self.args.cuda)
            else:
                self.grouping.append(mutiple.QGrouping(self.args, self.in_emb_dim)).to(self.args.cuda)

    def compute_dist(self, cur_feat):

        dist1 = compute_dist(self.args, cur_feat, cur_feat)
        max_dist = torch.max(dist1, dim=0)[0].unsqueeze(1)
        dist1 = torch.exp(dist1 - max_dist.repeat(1, dist1.size(1)))
        dist1 = torch.sum(dist1).mean()
        return dist1

    def compute_spatial(self, index_all, out_adj):
        spatial_corrdinates = sptial_neighbor_matrix(index_all, 3, out_adj)
        # dist2 可以理解为空间坐标的距离
        dist = compute_dist(self.args, spatial_corrdinates, spatial_corrdinates)
        dist = dist / torch.tile(torch.sqrt(torch.sum(dist ** 2, 1)), (dist.shape[0], 1))
        return dist
    def arpha(self, index_graph):
        # arpha = self.idx_graph[cluster](index_graph)
        index_graph_mean =list(index_graph)
        index_graph_mean = sum(index_graph_mean) /len(index_graph_mean)
        x = np.zeros(len(index_graph))
        for i in range(len(index_graph)):
            if(list(index_graph)[i]-index_graph_mean>0):
                x[i] = 1
            else:
                x[i] = 0
        index_graph = torch.FloatTensor(x).reshape(-1,1).to(self.args.cuda)
        return index_graph
    def to_cuda(self, x, device):
        if device:
            x = x.to(device)
        return x

