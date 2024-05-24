import math
import scipy.io as sio
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from sklearn.decomposition import PCA
from texttable import Texttable
from scipy.sparse import coo_matrix
# from sklearn.model_selection import train_test_split
import scipy.sparse as sp

from src.test import _split_with_min_per_class, _split_with_min_per_class3, _split_with_min_per_class1


def label_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    data = pd.read_csv(path,header=None)
    label = np.array(data.iloc[:, 1]).reshape(-1, 1)
    # print("target' type is"+type(target))
    return label


def encode_onehot(labels):
    classes = len(set(list(labels)))
    classes_dict = {c: np.identity(classes)[i, :] for i, c in
                    enumerate(set(list(labels)))}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
# 转换格式
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()

def aver(adj, hops, feature_list, alpha=0.15):
    input_feature = []
    for i in range(adj.shape[0]):
        hop = hops[i].int().item()
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
            print("fel",feature_list[0][i])
        else:
            fea = 0
            for j in range(hop):
                #  1-alpha 表示当前层的贡献，而 alpha 表示原始特征向量的贡献。最后将加权平均后的特征向量作为该节点的平滑后特征，并将其添加到 input_feature 列表中
                fea += (1-alpha)*feature_list[j][i].unsqueeze(0) + alpha*feature_list[0][i].unsqueeze(0)
            fea = fea / hop
        input_feature.append(fea)
    input_feature = torch.cat(input_feature, dim=0)
    return input_feature

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    # sorted是对字段以一种逻辑进行排序
    keys = sorted(args.keys())
    # 可视化打印
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph

# def feature_reader(path):
#     """
#     Reading the sparse feature matrix stored as csv from the disk.
#     :param path: Path to the csv file.
#     :return features: Dense matrix of features.
#     """
#     features = pd.read_csv(path)
#     node_index = features["node_id"].values.tolist()
#     feature_index = features["feature"].values.tolist()
#     feature_values = features["value"].values.tolist()
#     node_count = max(node_index)+1
#     feature_count = max(feature_index)+1
#     features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
#     print()
#     return features

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["label"]).reshape(-1,1)
    return target

def target_all_reader(path):
    # gnd = sio.loadmat(path)['Indian_pines_gt']
    # gnd = sio.loadmat(path)['Houston_gt']
    gnd = sio.loadmat(path)['PaviaU_sub_gt']
    gnd = np.array(gnd)
    target_all = gnd
    return target_all

def ind2sub1(array_shape, ind):
    # array_shape is array,an array of two elements where the first element is the number of rows in the matrix
    # and the second element is the number of columns.
    # ind is vector index with the python rule
    ind = np.array(ind.cpu().numpy())
    array_shape = np.array(array_shape)
    rows = (ind.astype('int') // array_shape[1].astype('int'))
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols
def ind2sub(array_shape, ind):
    array_shape = torch.tensor(array_shape)  # 将数组形状转换为 PyTorch 张量
    rows = ind // array_shape[1]  # 计算行索引
    cols = ind % array_shape[1]  # 计算列索引
    return rows, cols  # 返回行索引和列索引

def sptial_neighbor_matrix(index_all, neighbor, gt):
    """extract the spatial neighbor matrix, if x_j belong to x_i neighbors, thus S_ij = 1"""
    # index_all = np.concatenate((index_train_all, index_test))
    # index_all = dy11
    L_cor = torch.zeros([2, 1])
    for kkk in range(len(index_all)):
        [X_cor, Y_cor] = ind2sub([gt.shape[0], gt.shape[1]], index_all[kkk])
        XY_cor = torch.tensor([X_cor, Y_cor]).view(2, 1)
        L_cor = torch.cat((L_cor, XY_cor), dim=1)
    L_cor = L_cor[:, 1:]  # 删除第一列
    return torch.transpose(L_cor, 0, 1)


def compute_dist(args, array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array or torch tensor on GPU with shape [m1, n]
        array2: numpy array or torch tensor on GPU with shape [m2, n]
        type: one of ['cosine', 'euclidean']
    Returns:
        numpy array or torch tensor on GPU with shape [m1, m2]
    """
    device = args.cuda
    assert type in ['cosine', 'euclidean']
    if isinstance(array1, np.ndarray):
        array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
        array2 = torch.from_numpy(array2)
    if torch.cuda.is_available():
        array1 = array1.to(device)
        array2 = array2.to(device)
    if type == 'cosine':
        dist = torch.matmul(array1, array2.T)
        return dist
    else:
        square1 = torch.sum(torch.square(array1), dim=1).unsqueeze(1)
        square2 = torch.sum(torch.square(array2), dim=1).unsqueeze(0)
        t = -2 * torch.matmul(array1, array2.T)
        squared_dist = t + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = torch.sqrt(squared_dist)
        return dist


def compute_dist1(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """

    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        #array1 = normalize(array1, axis=1)
       # array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        # 平方后按列求和，
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        t = - 2 * np.matmul(array1, array2.T)
        squared_dist = t + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)#np.exp(dist - np.tile(np.max(dist, axis=0)[..., np.newaxis], np.size(dist, 1)))
        return dist
def train_test_spilts(args, class_count, clusters, sg_features, sg_targets, node_index, train_ratio):
    lists = np.zeros(int(class_count))   # 初始化
    sg_train_nodes={}
    sg_test_nodes={}
    all_data_nodes={}
    totol_sample_num = np.zeros(int(class_count))   # 初始化
    totol_test_num = np.zeros(int(class_count))     # 初始化
    # mask = np.zeros(int(class_count) )
    # mask_totol = np.zeros(int(class_count) )
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    for cluster in clusters:
        sg_train_nodes[cluster], sg_test_nodes[cluster], class_num, test_num = _split_with_min_per_class(X=sg_features[cluster], y=sg_targets[cluster], test_size=args.test_ratio, list=lists)
        print("子图{}训练集数量：{}".format(cluster, class_num))
        # print("子图{}测试集数量：{}".format(cluster, test_num))
        # print(type(all_data_nodes[cluster]),'\n', len(all_data_nodes[cluster]))
        # print(sg_train_nodes[cluster].shape, '\n', sg_test_nodes[cluster].shape)
        all_data_nodes[cluster] = np.array(sg_train_nodes[cluster]+sg_test_nodes[cluster])

        # 保存子图
        # save_figure(all_data_nodes[cluster], sg_features[cluster], sg_targets[cluster], cluster)
        sg_test_nodes[cluster] = sorted(sg_test_nodes[cluster])
        sg_train_nodes[cluster] = sorted(sg_train_nodes[cluster])
        sg_train_nodes[cluster] = torch.LongTensor(sg_train_nodes[cluster])
        sg_test_nodes[cluster] = torch.LongTensor(sg_test_nodes[cluster])
        totol_sample_num += class_num
        totol_test_num += test_num
    return sg_train_nodes, sg_test_nodes, totol_sample_num, totol_test_num

def Cal_accuracy(predict, label):
    estim_label = predict.argmax(1)
    # estim_label = estim_label.detach().cpu().numpy()
    # true_label = label.detach().cpu().numpy()
    true_label = label
    n = true_label.shape[0]
    OA = np.sum(estim_label == true_label) * 1.0 / n
    correct_sum = np.zeros((max(true_label) + 1))
    reali = np.zeros((max(true_label) + 1))
    predicti = np.zeros((max(true_label) + 1))
    producerA = np.zeros((max(true_label) + 1))

    predictions = []

    # 循环计算每个类别的预测结果
    for i in range(1, max(true_label) + 1):
        correct_sum[i] = np.sum(true_label[np.where(estim_label == i)] == i)
        reali[i] = np.sum(true_label == i)
        predicti[i] = np.sum(estim_label == i)
        producerA[i] = correct_sum[i] / reali[i]

        # 计算预测结果并添加到列表中
        predictions.append(producerA[i])

    # 计算预测结果的均值
    predictions_mean = np.mean(predictions)
    # print(producerA)
    Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))
    return OA, Kappa, predictions_mean

def DataLoader(dataset_name):
    data = []
    if(dataset_name =='IP'):
        print("Indian Pine")
        # data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/IndianPine/Indian_pines_corrected.mat')
        # data = data_mat['indian_pines_corrected']
        # gt_mat = sio.loadmat('/home/wjx/Cluster_Group/input/Indian_pines_gt.mat')
        # gt = gt_mat['indian_pines_gt']
        class_count = 16
        train_ratio = 0.015
        num_graph = 6

    if(dataset_name =='PU'):
        print("PaviaU")
        # data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/paviau/PaviaU.mat')
        # data = data_mat['paviaU']
        # gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/paviau/PaviaU_gt.mat')
        # gt = gt_mat['paviaU_gt']
        class_count = 10
        train_ratio = 0.005
        num_graph = 16

    if (dataset_name == 'HS'):
        print("Houston")
        # data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/houston/Houston.mat')
        # data = data_mat['Houston']
        # gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/houston/Houston_gt.mat')
        # gt = gt_mat['Houston_gt']
        class_count = 16
        train_ratio = 0.01
        num_graph = 6


    if (dataset_name == 'SA'):
        print("Salinas")
        # data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/salinas/Salinas_corrected.mat')
        # data = data_mat['salinas_corrected']
        # gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/salinas/Salinas_gt.mat')
        # gt = gt_mat['salinas_gt']
        class_count = 17
        train_ratio = 0.01
        num_graph = 30

    return class_count, train_ratio, num_graph


# def feature_reader(features):
#     """
#     对特征数据去噪
#     """
#     # 数据标准化
#     height, width, bands = features.shape
#     data = np.reshape(features, [height * width, bands])
#     if (bands > 60):  # 将数据映射到维度为60的空间
#         pca = PCA(n_components=60)
#         # 拟合数据并进行降维
#         data = pca.fit_transform(data)
#     minMax = preprocessing.StandardScaler()
#     data = minMax.fit_transform(data)  # 将数据转换为标准正态分布（均值为0,标准差为1）。通过从每个特征中减去样本均值并除以标准差来实现的。它常用于那些假定输入特征为正态分布的模型，如许多线性模型和神经网络。
#     # data = np.reshape(data, [height, width, bands])
#     return data  # 145*145, 60

def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path,header=None)
    # node_index = features["node_id"].values.tolist()
    # feature_index = features["feature_id"].values.tolist()
    # feature_values = features["value"].values.tolist()
    node_ids = features.iloc[:, 0].tolist()
    g_features = features.iloc[:, 1:].values
    node_count = max(node_ids)+1
    # feature_count = max(feature_index)+1
    # features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    return g_features

def edge_construction(self, train_node, node_index, train_node_rl):  # train_node 训练集节点；node_index：子图节点；
    # (self.train_node[i], self.sg_nodes[i], self.train_node[i])
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))  path="../data/cora/", dataset="cora",
    ran_num_sample = 2  # Indian_pines_PCA  PaviaU_sub Houston_PCA
    gnd = self.g_label.reshape(-1, 1)[node_index]  # 根据子图中的节点选取对应标签
    gt = np.squeeze(gnd)  # 去掉长度为1的维度
    features = self.CM.features[node_index]  # 选取节点特征

    print('this is features', np.array(features).shape)
    # dist1 计算特征间的距离，可以理解为光谱距离
    dist1 = compute_dist(self.args, np.array(features), np.array(features))
    # 将dist1中的所有值减去当前行的最大值。
    dist1 = torch.exp(dist1 - torch.max(dist1, dim=0, keepdim=True).values.expand_as(dist1))
    # dist1 = torch.exp(dist1 - torch.max(dist1, dim=0, keepdim=True).values.unsqueeze(0)).expand_as(dist1)
    # dist1 = np.exp(dist1 - np.tile(np.max(dist1, axis=0)[..., np.newaxis], np.size(dist1, 1)))
    spatial_corrdinates = sptial_neighbor_matrix(np.array(node_index), 3, self.g_label)
    # dist2 可以理解为空间坐标的距离
    dist2 = compute_dist(self.args, np.array(spatial_corrdinates), np.array(spatial_corrdinates))
    # dist2 = dist2 / np.tile(torch.sqrt(torch.sum(dist2 ** 2, 1)), (dist2.shape[0], 1))
    dist2 = dist2 / torch.sqrt(torch.sum(dist2 ** 2, dim=1, keepdim=True)).expand_as(dist2)
    # 这里的β=30
    dist = dist1 + 30 * dist2  # dist = dist1 + 30*dist2
    dist_new = dist

    idx_train = train_node_rl
    idx_train = np.array(idx_train).astype(int)
    # 初始化一个距离
    labeled_dist = np.zeros((1, len(idx_train)))
    for i in range(len(idx_train)):
        labeled_dist = np.concatenate(
            (labeled_dist, np.array(dist[idx_train[i], idx_train].cpu())[np.newaxis, ...]))
    # 删除第一行为0的dist
    labeled_dist = np.delete(labeled_dist, 0, axis=0)
    first_block_intra_dist = labeled_dist[0:ran_num_sample, 0:ran_num_sample]
    first_block_intra_dist_sum = np.sum(first_block_intra_dist)
    sum_2 = 0
    for j in range(np.max(gt).astype(int) - 1):
        j = j + 1
        sum_1 = labeled_dist[j * ran_num_sample:(j + 1) * ran_num_sample,
                j * ran_num_sample:(j + 1) * ran_num_sample]
        sum_2 = sum_2 + np.sum(sum_1)
    sum_intra_all = sum_2 + first_block_intra_dist_sum
    integer_intra = ran_num_sample * ran_num_sample * np.max(gt)
    average_sum = sum_intra_all / (integer_intra)
    # print('this is labels_no_zero', gt)
    # print('this is 类内平均', average_sum)
    average_inter = (np.sum(labeled_dist) - np.sum(sum_intra_all)) / (labeled_dist.shape[0] ** 2 - integer_intra)
    # print('this is 类间平均', average_inter)
    # print('这是类内类间初始差值（阈值）', average_inter - average_sum)
    # 这里是去除自己到自己的一个距离，再把大于类内均值的设为inf
    dist_new = dist_new.cpu() - np.diag(np.diag(dist_new.cpu()))
    dist_new[dist_new > average_sum] = float('inf')
    S_dist = np.exp(-dist_new / 6)  # 6
    lam = 0.01  # 5
    # 设置邻接矩阵的表达形式
    S = np.zeros((dist.shape[0], dist.shape[1]))
    # 先找到每一行的非0元素位置，得到值ai，再把dist矩阵中的对应元素赋给di，拿ai-di得到ad，将概率值设为1.
    for k in range(len(gt)):
        a0 = S_dist[k]
        idxa0 = np.where(a0 > 0)
        ai = a0[idxa0]
        di = dist[k][idxa0]
        ad = ai.to(self.device) - 0.5 * lam * di
        S[k][idxa0] = EProjSimplex_new(ad.cpu())
    adj = S
    # sio.savemat('../data/'+ self.args.Dataset_name + '/adj.mat', {'adj': adj})
    adj = torch.FloatTensor(adj)
    return adj

def EProjSimplex_new(v, k=1):
    v = np.matrix(v)
    ft = 1
    n = np.max(v.shape)

    if np.min(v.shape) == 0:
        return v, ft

    v0 = v - np.mean(v) + k / n

    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10 ** -10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft = ft + 1
            if ft > 100:
                v1[v1 < 0] = 0.0
                break
        x = v1.copy()
        x[x < 0] = 0.0
    else:
        x = v0
    return x