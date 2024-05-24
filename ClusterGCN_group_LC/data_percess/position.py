import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import networkx as nx
from torch.nn import ReLU
from tqdm import tqdm
from torch.nn.modules.linear import Linear
from utils import *
from sklearn.decomposition import PCA

output_path = "/home/wjx/Cluster_Group/output/"

class PositionEncoding(nn.Module):
    def __init__(self, args, graph):
        super(PositionEncoding, self).__init__()
        self.args = args
        self.device = self.args.cuda
        self.data_name = self.args.Dataset_Name
        self.dist_matrix: np.ndarray = None
        self.graph = graph
        self.node_cnt = len(self.graph)
        self.dropout = nn.Dropout()
        self.pca = self.args.pca_distance
        if self.pca:
            self.embedding = nn.Linear(self.args.pca_dim, self.args.position_dim)
            self.pca_dim = self.args.pca_dim
            self.pca = True
        else:
            self.embedding = nn.Linear(len(self.graph)+1, self.args.position_dim)

        self.init_dist_matrix()


    def init_dist_matrix(self):
        matrix_path = os.path.join('/home/wjx/Cluster_Group/output/', self.data_name)
        try:
            self.dist_matrix = np.load(os.path.join(matrix_path,'dist_matrix.npy'))
            print('Loading precomputed cosine postion encoding files……')
        except:
            os.makedirs(matrix_path, exist_ok=True)
            dist_path = os.path.join(matrix_path,'dist_matrix.npy')
            if os.path.exists(dist_path):
                print('Loading distance matrix form file…… ')
                self.dist_matrix = np.load(dist_path)
            else:
                print('Calculating shortest paths...')
                self.dist_matrix = np.zeros([self.node_cnt+1, self.node_cnt+1])   # 初始化一个N*N的矩阵C

                for i, values in tqdm(nx.all_pairs_shortest_path_length(self.graph)):   # nx.all_pairs_shortest_path_length(self.graph)计算每对节点之间的最短距离
                    for j, length in values.items():
                        self.dist_matrix[i+1, j+1] = length

                np.save(os.path.join(matrix_path, 'dist_matrix.npy'), self.dist_matrix)  # 最短距离矩阵保存到dist_matrix.npy文件中
            '''
            公式（2）
            '''
            self.dist_matrix /= np.nanmax(self.dist_matrix,axis=0, keepdims=True) # np.nanmax——最大值计算。对dist_matrix中每个元素进行归一化处理，让所有的距离值都缩放到 [0, 1]
            self.dist_matrix = np.cos(self.dist_matrix * np.pi)   # cos(norm(C) * π)
            self.dist_matrix[np.isnan(self.dist_matrix)] = - 1.5   # 对不可达节点将其元素设为-1.5
            '''
            公式（2）
            '''
            if(len(self.dist_matrix) == len(self.graph)):    # 检查dist_matrix与图的节点长度是否相等
                self.dist_matrix = np.vstack((np.zeros((1,self.node_cnt)),self.dist_matrix))        # np.vstack等价于np.concatenate((a,b),axis = 0)，在dist_matrix上拼一行全0数组
                self.dist_matrix = np.hstack((np.zeros((self.node_cnt+1,1)),self.dist_matrix))   # np.hstack等效于 np.concatenate((a,b),axis = 1)，在dist_matrix上拼一列全0数组
                print('Saving padded cosine distance matrix to', os.path.join(matrix_path,'cosine_matrix.npy'))
                np.save(os.path.join(matrix_path,'cosine_matrix.npy'), self.dist_matrix)    # 将余弦距离矩阵保存到 cosine_matrix.npy文件中
        print('Phase propagation finished')
        #if hyper_params['device'] == 'cuda':
        #    self.dist_matrix = self.dist_matrix.cuda()
        '''
        公式（3）
        '''
        if self.pca:
            if os.path.exists(os.path.join(matrix_path, 'pca_dist.npy')):
                print('Loading PCA from', os.path.join(matrix_path,'pca_dist.npy'))
                self.pca_matrix = np.load(os.path.join(matrix_path, 'pca_dist.npy'))
            else:
                print('Pre-computing PCA')
                self.pca = PCA(n_components= self.args.pca_dim)
                self.pca_matrix = self.pca.fit_transform(self.dist_matrix)       # 调用PCA.fit_transform方法，用dist_matrix来训练PCA模型，实现数据降噪降维
                np.save(os.path.join(matrix_path, 'pca_dist.npy'), self.pca_matrix)
                print('Saving PCA to', os.path.join(matrix_path, 'pca_dist.npy'))   # pca_dist保存到pca_dist.npy中
            tmp_matrix = torch.zeros([len(self.graph)+1, self.pca_dim])    # 初始化N*pca_dim矩阵
            tmp_matrix[1:] = torch.from_numpy(self.pca_matrix).float()[1:]  # 将pca_matrix填充到tmp_matrix中，第一行为空
        else:
            tmp_matrix = torch.zeros([len(self.graph)+1, len(self.graph)+1])
            tmp_matrix[1:, 1:] = torch.from_numpy(self.pca_matrix).float()[1:, 1:]
        self.pca_matrix = tmp_matrix   # 将tmp_matrix赋值给pca_matrix
        self.pca_matrix /= self.pca_matrix.std()   # self.pca_matrix.std()计算pca_matrix的标准差，将self.pca_matrix的每个元素除以整个矩阵的标准差来进行标准化。
        self.pca_matrix.to(self.device)
        '''
        公式（3）
        '''

    def forward(self, nodes):
        nodes = torch.tensor(nodes)
        nodes_flat = nodes.reshape(-1)
        pe = self.pca_matrix[nodes_flat].reshape(nodes.shape[0], nodes.shape[1], -1).float().to(self.device)
        out = self.dropout(self.embedding(pe))
        return out

class LSTM_pooling(nn.Module):
    def __init__(self, args):
        super(LSTM_pooling, self).__init__()
        self.args = args
        # self.pooling = nn.LSTM(input_size=self.args.position_dim,
        #                        hidden_size=int(hyper_params['hidden2_dim']/2),
        #                        num_layers=2, batch_first=True, dropout = 0.5, bidirectional=True)
        self.pooling = nn.LSTM(input_size=self.args.position_dim,
                               hidden_size=int(self.args.hid_dim),
                               num_layers=2, batch_first=True, dropout = 0.5, bidirectional=True)
        # self.base_gcn = GraphConvSparse(hyper_params['hidden1_dim'], hyper_params['hidden1_dim'])
        self.dropout = nn.Dropout()  # hidden_last

    def forward(self, z):
        lstm_out, (hidden, c_n) = self.pooling(z)
        return lstm_out

    def neighbor_pooling(self, node_embedding, adj_norm):
        node_embedding = self.base_gcn(node_embedding, adj_norm)
        node_embedding = self.dropout(node_embedding)
        return node_embedding

class MLP(nn.Module):
    def __init__(self, in_features, hid_dim, bias=True):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features, hid_dim*2)
        self.linear2 = nn.Linear(hid_dim*2, hid_dim)
        self.relu = ReLU()

    def forward(self, x):
        # 通过第一层全连接层和激活函数
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x