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

from src_group.test import _split_with_min_per_class, _split_with_min_per_class3, _split_with_min_per_class1


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
    edges = pd.read_csv(path)
    # # # nx.from_pandas_edgelist() 函数从 Pandas DataFrame 创建图形。edge_attr='e_fet' 参数将 e_fet 列中的值作为边的权重添加到图中。
    # # # create_using=nx.DiGraph() 参数指定创建一个有向图对象。
    # # # create_using 参数的值被更改为 nx.Graph()，表示将创建一个无向图对象。
    # edge_source = edges["id1"].values.tolist()
    # edge_target = edges["id2"].values.tolist()
    # edge_fea1 = edges["e_fet"].values.tolist()
    # edge_fea = coo_matrix((edge_fea1, (edge_source, edge_target)), shape=(edges.shape[0], 1)).toarray()
    # graph = nx.from_pandas_edgelist(edges, source='id1', target='id2', edge_attr='e_fet', create_using=nx.Graph())
    # from_edgelist返回一个由列表中元素构成的图形
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
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/input/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/input/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        class_count = 17
        train_ratio = 0.015
        num_graph = 6

    if(dataset_name =='PU'):
        data_mat = sio.loadmat('../data/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('../data/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        class_count = 10
        train_ratio = 0.005
        num_graph = 12

    if (dataset_name == 'HS'):
        data_mat = sio.loadmat('../data/Houston.mat')
        data = data_mat['houston']
        gt_mat = sio.loadmat('../data/Houston_gt.mat')
        gt = gt_mat['houston_gt']
        class_count = 16
        train_ratio = 0.01
        num_graph = 6

    return data, gt, class_count, train_ratio, num_graph


def feature_reader(features):
    """
    对特征数据去噪
    """
    # 数据标准化
    height, width, bands = features.shape
    data = np.reshape(features, [height * width, bands])
    if (bands > 60):  # 将数据映射到维度为60的空间
        pca = PCA(n_components=60)
        # 拟合数据并进行降维
        data = pca.fit_transform(data)
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)  # 将数据转换为标准正态分布（均值为0,标准差为1）。通过从每个特征中减去样本均值并除以标准差来实现的。它常用于那些假定输入特征为正态分布的模型，如许多线性模型和神经网络。
    # data = np.reshape(data, [height, width, bands])
    return data  # 145*145, 60

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})