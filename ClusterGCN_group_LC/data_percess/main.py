import os
import pickle
import time
import scipy.io as sio
import networkx as nx
import numpy as np
import pandas as pd
import torch
from termcolor import cprint

from src.parser import parameter_parser
from clustering import ClusteringMachine
# from clustergcn import ClusterGCNTrainer
from utils import tab_printer, feature_reader, DataLoader, edge_construction, build_graph, graph_reader, \
    save_feature_gt, DataLoader1
from position import PositionEncoding
from tqdm import tqdm
from sklearn.decomposition import PCA

def init_dist_matrix(args, graph, node_index_rl):
    node_cnt = len(graph)
    matrix_path = os.path.join('/home/wjx/Cluster_Group/input/', args.Dataset_Name)
    try:
        dist_matrix = np.load(os.path.join(matrix_path, 'dist_matrix.npy'))
        print('Loading precomputed cosine postion encoding files……')
    except:
        os.makedirs(matrix_path, exist_ok=True)
        dist_path = os.path.join(matrix_path, 'dist_matrix.npy')
        if os.path.exists(dist_path):
            print('Loading distance matrix form file…… ')
            dist_matrix = np.load(dist_path)
        else:
            print('Calculating shortest paths...')
            dist_matrix = np.zeros([node_cnt + 1, node_cnt + 1])  # 初始化一个N*N的矩阵C

            for i, values in tqdm(nx.all_pairs_shortest_path_length(graph)):  # nx.all_pairs_shortest_path_length(self.graph)计算每对节点之间的最短距离
                for j, length in values.items():
                    dist_matrix[node_index_rl[i], node_index_rl[j]] = length
            np.save(os.path.join(matrix_path, 'dist_matrix.npy'), dist_matrix)  # 最短距离矩阵保存到dist_matrix.npy文件中
        '''
        公式（2）
        '''
        dist_matrix /= np.nanmax(dist_matrix, axis=0, keepdims=True)  # np.nanmax——最大值计算。对dist_matrix中每个元素进行归一化处理，让所有的距离值都缩放到 [0, 1]
        dist_matrix = np.cos(dist_matrix * np.pi)  # cos(norm(C) * π)
        dist_matrix[np.isnan(dist_matrix)] = - 1.5  # 对不可达节点将其元素设为-1.5
        '''
        公式（2）
        '''
        if (len(dist_matrix) == len(graph)):  # 检查dist_matrix与图的节点长度是否相等
            dist_matrix = np.vstack((np.zeros((1, node_cnt)),dist_matrix))  # np.vstack等价于np.concatenate((a,b),axis = 0)，在dist_matrix上拼一行全0数组
            dist_matrix = np.hstack((np.zeros((node_cnt+1, 1)),dist_matrix))  # np.hstack等效于 np.concatenate((a,b),axis = 1)，在dist_matrix上拼一列全0数组
            print('Saving padded cosine distance matrix to', os.path.join(matrix_path, 'cosine_matrix.npy'))
            np.save(os.path.join(matrix_path, 'cosine_matrix.npy'), dist_matrix)  # 将余弦距离矩阵保存到 cosine_matrix.npy文件中
    print('Phase propagation finished')
    # if hyper_params['device'] == 'cuda':
    #    self.dist_matrix = self.dist_matrix.cuda()
    '''
    公式（3）
    '''
    # pca = args.pca
    # pca_matrix =
    if args.pca:
        if os.path.exists(os.path.join(matrix_path, 'pca_dist.npy')):
            print('Loading PCA from', os.path.join(matrix_path, 'pca_dist.npy'))
            pca_matrix = np.load(os.path.join(matrix_path, 'pca_dist.npy'))
        else:
            print('Pre-computing PCA')
            pca = PCA(n_components=args.pca_dim)
            pca_matrix = pca.fit_transform(dist_matrix)  # 调用PCA.fit_transform方法，用dist_matrix来训练PCA模型，实现数据降噪降维
            np.save(os.path.join(matrix_path, 'pca_dist.npy'), pca_matrix)
            print('Saving PCA to', os.path.join(matrix_path, 'pca_dist.npy'))  # pca_dist保存到pca_dist.npy中
        tmp_matrix = torch.zeros([len(graph) + 1, args.pca_dim])  # 初始化N*pca_dim矩阵
        tmp_matrix[1:] = torch.from_numpy(pca_matrix).float()[1:]  # 将pca_matrix填充到tmp_matrix中，第一行为空
    else:
        tmp_matrix = torch.zeros([len(graph) + 1, len(graph) + 1])
        tmp_matrix[1:, 1:] = torch.from_numpy(dist_matrix).float()[1:, 1:]   # 不进行pca的距离矩阵
    pca_matrix = tmp_matrix  # 将tmp_matrix赋值给pca_matrix
    pca_matrix /= pca_matrix.std()  # self.pca_matrix.std()计算pca_matrix的标准差，将self.pca_matrix的每个元素除以整个矩阵的标准差来进行标准化。
    # pca_matrix.to(args.cuda)
    '''
    公式（3）
    '''


def main(args):
    # 设置随机数种子，保证实验结果的可再现
    # torch.manual_seed(args.seed)
    # 可视化参数表制作
    # tab_printer(args)
    # 读取边索引
    # base_graph = os.path.join('model_dat', hyper_params['data_name']), exist_ok=True)
    device = args.cuda
    node_index_rl = {}
    graph_path = os.path.join(args.graph_path, args.Dataset_Name,"edge_list.csv")
    features, lbls, G_adj, index_all = DataLoader(args.Dataset_Name)
    # # 对聚类器做一个初始化，初始化数据
    # CM = ClusteringMachine(args, features, lbls, class_count)
    print('building base graph……')
    # g_feature, gt_index, node_index = CM.sub_generation(args, num_graph=1)  # 构建base-graph，得到特征、标签及对应节点id
    # # node_index_rl = gt_index
    try:
        base_graph = graph_reader(graph_path)
        cprint('Loading base graph file……',color="green")
    except:
        cprint("building adj to adj.mat……",color="green")
        sio.savemat('/home/wjx/Cluster_Group/input/IP/' + '/adj.mat', {'adj': G_adj})
        G_adj = torch.FloatTensor(G_adj)
        cprint("building base-graph from adj",color="green")
        base_graph = build_graph(graph_path, G_adj, index_all)
        print("saving to edge_list.csv……")
        # nx.write_edgelist(base_graph, graph_path, data=False)
    print("saving feature and labels to files")
    save_feature_gt(features, index_all, lbls)
    node_index_rl = {node: i for i, node in enumerate(index_all)}
    init_dist_matrix(args, base_graph, node_index_rl)   # 计算距离矩阵



if __name__ == "__main__":
    """
    Parsing command line parameters, reading   data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # 配置参数
    args = parameter_parser()
    # args.embedding_dim = [60 * i for i in args.num_group]
    main(args)
