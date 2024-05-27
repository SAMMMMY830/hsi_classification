import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from modules import norm_layer
from src_group import graphlearn
from src_group.utils import sptial_neighbor_matrix, compute_dist
from graph_learn import learn_graph
from graphlearn import GraphLearner
from GCNlayer import GCNLayers,MLP
from para_model import Para_model
import Outils.multiple_embedding as mutiple

class DeeperGCN(torch.nn.Module):

    def __init__(self, args, node_feat_dim, hid_dim, out_dim, norm='batch', beta=1.0, clusters = 0, degree_max=0):
        super(DeeperGCN, self).__init__()
        # para_init

        self.args = args
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

        self.gcns.append(GCNLayers(node_feat_dim, node_feat_dim))
        self.norms.append(norm_layer(norm,  hid_dim))
        self.d_norms.append(norm_layer(norm,  hid_dim))
        self.linear.append(nn.Linear(node_feat_dim, hid_dim))

        self.in_emb_dim = self.module_num * self.hid_dim
        if args.grouping_layer == 'mul_linear':
            self.grouping.append(mutiple.MLGrouping(args, self.in_emb_dim))
        else:
            self.grouping.append(mutiple.QGrouping(args, self.in_emb_dim))
        self.mlp.append(MLP(self.args.embedding_dim[0], self.hid_dim))
        self.lc = nn.Linear(self.hid_dim, self.class_num)

        for cluster in self.clusters:
            self.embedding.append(nn.Embedding(len(degree_max[cluster]), hid_dim).to(self.args.cuda))
            self.idx_graph.append(nn.Embedding(len(degree_max[cluster]), hid_dim))
            self.graph_learner.append(GraphLearner(node_feat_dim, hid_dim, device=self.args.cuda, epsilon= self.args.epsilon))
            # self.para_model.append(Para_model().to(self.args.cuda))
        # self.embedding.append(nn.Embedding(, hid_dim))


    def forward(self, node_feats, train_nodes, cluster, mode, degeree, layer, init_adj, target):
        init_feat = node_feats
        layer = int(layer)
        # 获取标签类别的数量
        num_classes = self.class_num

        # 将标签转换为 one-hot 编码矩阵
        device = target.device
        one_hot_matrix = torch.eye(num_classes).to(device)[target.flatten()]

        # 打印转换后的one-hot编码矩阵

        dist = 0
        if mode == 'train':
            # loss, layer = self.train_epoch1(node_feats, train_nodes, init_adj, layer, target, init_feat, cluster)
            x_group,x_n,totol_loss, layer = self.train_epoch2(node_feats, train_nodes, init_adj, layer, target, init_feat, cluster, one_hot_matrix)
            return x_group[layer],x_n[layer],totol_loss, layer
        else:
            output = self.test_epoch2(node_feats, init_adj, layer, cluster)
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
        # temp = torch.mm(features.transpose(-1, -2), torch.mm(L, features))
        # temp2 = 10 * torch.mm(dist.transpose(-1, -2), torch.mm(L, dist))
        # graph_loss += self.args.smoothness_ratio * (torch.trace(temp)+torch.trace(temp2)) / int(np.prod(out_adj.shape))
        graph_loss += self.args.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        # ones_vec = self.to_cuda(torch.ones(out_adj.size(-1)), self.args.cuda)

        # self.config['degree_ratio']为b    self.config['sparsity_ratio']为γ，下方这两项为f(A)
        # graph_loss += -self.args.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + self.args.min_num)).squeeze() / out_adj.shape[-1]
        graph_loss += self.args.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def get_degree(self, adj):
        d = adj.sum(0).long()
        return d

    def train_epoch2(self, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster, dist):
        # init_adj = self.normalize_adj(init_adj)
        xs = []
        x_pool = []
        x_group = []
        x_n = []
        node_num = node_feats.size(0)
        pre_feat = node_feats
        # cur_raw_adj经过掩码处理过的， cur_adj加上了原始的图
        cur_feat = self.norms[layer](pre_feat)
        cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster)
        d = self.get_degree(cur_adj)
        d = self.embedding[cluster](d)
        cur_feat = F.relu(cur_feat)
        cur_feat = F.dropout(cur_feat, p=self.dropout, training=self.training)
        cur_feat = self.gcns[layer](cur_feat, cur_adj)
        # pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
        xs.append(pre_feat)
        x_n.append(torch.cat(xs, dim=-1))
        x_group.append(self.grouping[layer](x_n[layer]))
        x_cat = x_group[layer][0].reshape(node_num, -1)
        pre_feat = self.mlp[layer](x_cat)
        pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
        x_pool.append(pre_feat + cur_feat)
        # cur_feat = cur_feat + pre_feat
        # 增加网络
        self.add_network(layer, cur_feat)
        # 计算距离
        best_dist = self.compute_dist(cur_feat)
        output = torch.nn.functional.log_softmax(cur_feat, dim=1)
        loss1 = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes])
        loss1 = loss1 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
        # 循环准备
        first_adj = cur_adj
        totol_loss = loss1
        while (1):
            xs = []
            layer += 1
            pre_feat = cur_feat
            cur_feat = self.norms[layer](pre_feat)
            cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster)
            d = self.get_degree(cur_adj)
            d = self.embedding[cluster](d)
            cur_feat = F.relu(cur_feat)
            cur_feat = F.dropout(cur_feat, p=self.dropout, training=self.training)
            cur_feat = self.gcns[layer](cur_feat, cur_adj)
            # pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
            xs.append(pre_feat)
            x_n.append(torch.cat(xs, dim=-1))
            x_group.append(self.grouping[layer](x_n[layer]))
            x_cat = x_group[layer][0].reshape(node_num, -1)
            pre_feat = self.mlp[layer](x_cat)
            pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
            x_pool.append(pre_feat + cur_feat)
            cur_feat = x_pool[layer]
            # cur_feat = cur_feat + pre_feat
            cur_dist = self.compute_dist(cur_feat)
            if cur_dist < best_dist:
                best_dist = cur_dist
                output = torch.nn.functional.log_softmax(cur_feat, dim=1)
                loss2 = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes])
                loss2 = loss2 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
                totol_loss +=loss2
                self.add_network(layer, cur_feat)
            else:
                layer -= 1
                output = torch.nn.functional.log_softmax(pre_feat, dim=1)
                loss = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes])
                loss2 = loss + self.add_graph_loss(cur_raw_adj, init_feat, dist)
                totol_loss = (totol_loss + loss2)/ (layer+1)
                break
        return x_group,x_n,totol_loss, layer

    def test_epoch2(self, node_feats, init_adj, layer, cluster):
        xs= []
        pre_feat = node_feats
        # cur_raw_adj经过掩码处理过的， cur_adj加上了原始的图
        cur_feat = self.norms[0](pre_feat)
        cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster, training=False)

        cur_feat = F.relu(cur_feat)
        cur_feat = F.dropout(cur_feat, p=self.dropout, training=False)
        cur_feat = self.gcns[0](cur_feat, cur_adj)

        d = self.get_degree(cur_adj)
        d = self.embedding[cluster](d)

        xs.append(pre_feat)
        x_n = torch.cat(xs, dim=-1)
        x_group = self.grouping[0](x_n)
        x_group = F.normalize(x_group[0], dim=-1)
        x_cat = x_group.reshape(node_feats.size(0), -1)
        pre_feat = self.mlp[0](x_cat)
        pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
        cur_feat = cur_feat + pre_feat
        # 循环准备
        first_adj = cur_adj
        for layers in range(layer):
            xs = []
            pre_feat = cur_feat
            cur_feat = self.norms[layers + 1](pre_feat)
            cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster, training=False)

            cur_feat = F.relu(cur_feat)
            cur_feat = F.dropout(cur_feat, p=self.dropout, training=False)
            cur_feat = self.gcns[layers + 1](cur_feat, cur_adj)

            d = self.get_degree(cur_adj)
            d = self.embedding[cluster](d)
            # pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
            xs.append(pre_feat)
            x_n = torch.cat(xs, dim=-1)
            x_group = self.grouping[layers + 1](x_n)
            x_group = F.normalize(x_group[layers + 1], dim=-1)
            x_cat = x_group.reshape(node_feats.size(0), -1)
            pre_feat = self.mlp[layers + 1](x_cat)
            pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
            cur_feat = cur_feat + pre_feat
            # pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
            # x_pool.append(pre_feat + cur_feat)
            # cur_feat = self.mlp[0](x_cat) + cur_feat
            # cur_feat = cur_feat + pre_feat
        output = torch.nn.functional.log_softmax(cur_feat, dim=1)
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

