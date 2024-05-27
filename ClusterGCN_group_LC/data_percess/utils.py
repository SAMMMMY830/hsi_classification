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
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.segmentation import slic,mark_boundaries
from src.test import _split_with_min_per_class, _split_with_min_per_class3, _split_with_min_per_class1


def save_feature_gt(g_feature, node_index, lbls):
    node_rl = list(range(len(node_index)))
    node_ids = node_index
    with open('/home/wjx/Cluster_Group/input/IP/features.csv', 'w', newline='') as fea_file:
        writer = csv.writer(fea_file)
        # writer.writerow(['NodeID', 'feature_id', 'value'])  # 写入表头
        # 写入每个节点的特征，转换 tensor 为 list
        for node_index, node_id, feature in tqdm(zip(node_rl, node_ids, g_feature.tolist()), total=len(node_ids),desc="Writing features CSV"):
            writer.writerow([node_index] + [node_id] + feature)

    with open('/home/wjx/Cluster_Group/input/IP/gt.csv', 'w', newline='') as gt_file:
        writer = csv.writer(gt_file)
        for node_index, node_id, gt_index in tqdm(zip(node_rl, node_ids, lbls.tolist()), total=len(node_ids), desc="Writing labels CSV"):
            writer.writerow([node_index] + [node_id] + [gt_index])


def edge_construction(args, node_index, orig_label, g_feature):
    # G_adj = edge_construction(args, node_index, orig_label, g_feature[0], gt_index[0])# 索引
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))  path="../data/cora/", dataset="cora",
    ran_num_sample = 5  # Indian_pines_PCA  PaviaU_sub Houston_PCA
    gnd = orig_label.reshape(-1, 1)[node_index]  # 根据子图中的节点选取对应标签
    gt = np.squeeze(gnd)  # 去掉长度为1的维度
    features = g_feature[node_index]  # 选取节点特征
    print('this is features', np.array(features).shape)
    # G_adj = edge_construction(args, node_index, orig_label, g_feature[0], gt_index[0])
    # dist1 可以理解为光谱距离
    dist1 = compute_dist(args, np.array(features), np.array(features))
    # 将dist1中的所有值减去当前行的最大值。
    dist1 = torch.exp(dist1 - torch.max(dist1, dim=0, keepdim=True).values.expand_as(dist1))
    # dist1 = torch.exp(dist1 - torch.max(dist1, dim=0, keepdim=True).values.unsqueeze(0)).expand_as(dist1)
    # dist1 = np.exp(dist1 - np.tile(np.max(dist1, axis=0)[..., np.newaxis], np.size(dist1, 1)))
    spatial_corrdinates = sptial_neighbor_matrix(np.array(node_index), 3, orig_label)
    # dist2 可以理解为空间坐标的距离
    dist2 = compute_dist(args, np.array(spatial_corrdinates), np.array(spatial_corrdinates))
    # dist2 = dist2 / np.tile(torch.sqrt(torch.sum(dist2 ** 2, 1)), (dist2.shape[0], 1))
    dist2 = dist2 / torch.sqrt(torch.sum(dist2 ** 2, dim=1, keepdim=True)).expand_as(dist2)
    # 这里的β=30
    dist = dist1 + args.edge_lam * dist2  # IP 20，PU 0.1，houston 0.3
    dist_new = dist
    idx_train = list(range(len(node_index)))
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
        ad = ai.to(args.cuda) - 0.5 * lam * di
        S[k][idxa0] = EProjSimplex_new(ad.cpu())
    # 生成邻接矩阵并写入文件
    adj = S
    print("adj.mat")
    sio.savemat('/home/wjx/Cluster_Group/input/IP/' + '/adj.mat', {'adj': adj})
    # sio.savemat('/home/wjx/Cluster_Group/input/IP/' + '/gt.mat', {'adj': adj})
    adj = torch.FloatTensor(adj)
    return adj


def build_graph(graph_path, adj, node_index):
    # G = nx.Graph()
    # for idx in node_index:
    #     G.add_node(idx)
    # 为图添加边
    # for i in tqdm(range(len(adj)), desc='Processing nodes'):
    #     for j in range(len(adj)):
    #         if adj[i][j] != 0:
    #             G.add_edge(i, j, weight=adj[i][j])

    # nx.draw(G, with_labels=True)
    # plt.show()
    # 添加边
    # for i in tqdm(range(len(adj)), desc= "Processing nodes"):
    #     for j in range(i + 1, len(adj)):
    #         if adj[i][j] != 0:  # 如果 i 和 j 之间存在边
    #             G.add_edge(node_index[i], node_index[j], weight=adj[i][j])
    adj = np.array(adj)
    G = nx.from_numpy_array(adj)

    mapping = {i: node_index[i] for i in range(len(node_index))}   # seq:id
    G = nx.relabel_nodes(G, mapping)
    # 将图转换为邻接表
    adj_id_list = nx.to_dict_of_lists(G)
    adj_list_id_df = pd.DataFrame([(k, v) for k, values in adj_id_list.items() for v in values], columns=["id1", "id2"])
    adj_list_id_df.to_csv(graph_path, index=False)
    return G


"""
=======================================================================================================
"""


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
            print("fel", feature_list[0][i])
        else:
            fea = 0
            for j in range(hop):
                #  1-alpha 表示当前层的贡献，而 alpha 表示原始特征向量的贡献。最后将加权平均后的特征向量作为该节点的平滑后特征，并将其添加到 input_feature 列表中
                fea += (1 - alpha) * feature_list[j][i].unsqueeze(0) + alpha * feature_list[0][i].unsqueeze(0)
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
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
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
    target = np.array(pd.read_csv(path)["label"]).reshape(-1, 1)
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


# def sptial_neighbor_matrix(index_all, neighbor, gt):
#     """extract the spatial neighbor matrix, if x_j belong to x_i neighbors, thus S_ij = 1"""
#     # index_all = np.concatenate((index_train_all, index_test))
#     # index_all = dy11
#     L_cor = torch.zeros([2, 1])
#     for kkk in range(len(index_all)):
#         [X_cor, Y_cor] = ind2sub([gt.shape[0], gt.shape[1]], index_all[kkk])
#         XY_cor = torch.tensor([X_cor, Y_cor]).view(2, 1)
#         L_cor = torch.cat((L_cor, XY_cor), dim=1)
#     L_cor = L_cor[:, 1:]  # 删除第一列
#     return torch.transpose(L_cor, 0, 1)

def sptial_neighbor_matrix(index_all, gt):
    """ extract the spatial coordinates(x,y) """

    L_cor = np.zeros([2, 1])
    for kkk in range(len(index_all)):
        [X_cor, Y_cor] = ind2sub([np.size(gt, 0), np.size(gt, 1)], index_all[kkk])
        XY_cor = np.array([X_cor, Y_cor])[..., np.newaxis]
        L_cor = np.concatenate((L_cor, XY_cor), axis=1)
    L_cor = np.delete(L_cor, 0, axis=1)  # delete 0 column

    return L_cor.transpose()


# def compute_dist(args, array1, array2, type='euclidean'):
#     """Compute the euclidean or cosine distance of all pairs.
#     Args:
#         array1: numpy array or torch tensor on GPU with shape [m1, n]
#         array2: numpy array or torch tensor on GPU with shape [m2, n]
#         type: # oneof ['cosine', 'euclidean']
#     Returns:
#         numpy array or torch tensor on GPU with shape [m1, m2]
#     """
#     device = args.cuda
#     assert type in ['cosine', 'euclidean']
#     if isinstance(array1, np.ndarray):
#         array1 = torch.from_numpy(array1)
#     if isinstance(array2, np.ndarray):
#         array2 = torch.from_numpy(array2)
#     if torch.cuda.is_available():
#         array1 = array1.to(device)
#         array2 = array2.to(device)
#     if type == 'cosine':
#         dist = torch.matmul(array1, array2.T)
#         return dist
#     else:
#         square1 = torch.sum(torch.square(array1), dim=1).unsqueeze(1)
#         square2 = torch.sum(torch.square(array2), dim=1).unsqueeze(0)
#         t = -2 * torch.matmul(array1, array2.T)
#         squared_dist = t + square1 + square2
#         squared_dist[squared_dist < 0] = 0
#         dist = torch.sqrt(squared_dist)
#         return dist


def compute_dist(array1, array2, type='euclidean'):
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
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)#np.exp(dist - np.tile(np.max(dist, axis=0)[..., np.newaxis], np.size(dist, 1)))
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
        # array1 = normalize(array1, axis=1)
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
        dist = np.sqrt(squared_dist)  # np.exp(dist - np.tile(np.max(dist, axis=0)[..., np.newaxis], np.size(dist, 1)))
        return dist


def train_test_spilts(args, class_count, clusters, sg_features, sg_targets, node_index, train_ratio):
    lists = np.zeros(int(class_count))  # 初始化
    sg_train_nodes = {}
    sg_test_nodes = {}
    all_data_nodes = {}
    totol_sample_num = np.zeros(int(class_count))  # 初始化
    totol_test_num = np.zeros(int(class_count))  # 初始化
    # mask = np.zeros(int(class_count) )
    # mask_totol = np.zeros(int(class_count) )
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    for cluster in clusters:
        sg_train_nodes[cluster], sg_test_nodes[cluster], class_num, test_num = _split_with_min_per_class(
            X=sg_features[cluster], y=sg_targets[cluster], test_size=args.test_ratio, list=lists)
        print("子图{}训练集数量：{}".format(cluster, class_num))
        # print("子图{}测试集数量：{}".format(cluster, test_num))
        # print(type(all_data_nodes[cluster]),'\n', len(all_data_nodes[cluster]))
        # print(sg_train_nodes[cluster].shape, '\n', sg_test_nodes[cluster].shape)
        all_data_nodes[cluster] = np.array(sg_train_nodes[cluster] + sg_test_nodes[cluster])

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
    if (dataset_name == 'IP'):
        print("Indian Pine")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/IndianPine/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/IndianPine/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        # class_count = 17
        # train_ratio = 0.015
        # num_graph = 6

    if (dataset_name == 'PU'):
        print("PaviaU")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/paviau/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/paviau/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        # class_count = 10
        # train_ratio = 0.005
        # num_graph = 16

    if (dataset_name == 'HS'):
        print("Houston")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/houston/Houston.mat')
        data = data_mat['Houston']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/houston/Houston_gt.mat')
        gt = gt_mat['Houston_gt']
        # class_count = 16
        # train_ratio = 0.01
        # num_graph = 6

    if (dataset_name == 'SA'):
        print("Salinas")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/salinas/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/salinas/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        # class_count = 17
        # train_ratio = 0.01
        # num_graph = 30

    gnd = np.array(gt)
    gnd = np.transpose(gnd)

    gnd = np.array(gnd)
    gt = gnd
    data_all = np.array(data)
    data_all = (data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all))  # 对数据进行标准化
    # img = io.imread("D:\yunding\paper_four\HSI_data\Indian_29_42_89.png")
    data_seg = slic(data_all, n_segments=2000, compactness=0.1)  # indian slic 分割 1500, 0.01; pavia 分割 1500, 1
    # out = mark_boundaries(img, data_seg)
    # plt.imshow(out)
    # plt.show()
    # sio.savemat('data_seg.mat', {'data_seg': data_seg})
    print('this is data shape', data_all.shape)

    row_gnd = np.size(gnd, 0)
    col_gnd = np.size(gnd, 1)
    dim = np.size(data_all, 2)
    gnd = np.reshape(gnd, (1, row_gnd * col_gnd))  # ground truth
    ground_truth = gnd.copy()
    fea = np.reshape(data_all, (1, row_gnd * col_gnd, dim))
    fea = np.squeeze(fea)
    # kmeans = KMeans(n_clusters=100, random_state=0).fit(fea)
    # data_seg = kmeans.labels_

    gnd = np.squeeze(gnd)
    ground_truth = gnd.copy()
    choice = np.where(gnd != 0)   # 标签不为0节点id（节点索引）
    # gnd_new = gnd[choice]
    segments = np.reshape(data_seg, (1, row_gnd * col_gnd))
    segments = np.squeeze(segments)

    index_all = choice
    fts = fea[choice]  # fts = samples*dimension
    lbls = gnd[choice]  # find labels
    seg_labels = segments[choice]
    unique_seg_lbls = np.unique(seg_labels)
    len_seg_lbls = len(unique_seg_lbls)  # find the unique length of segment labels
    mean_pixel = np.zeros((1, dim))
    unique_seg_labels = np.unique(seg_labels)
    max_seg_lab = np.max(segments)

    # YY = np.zeros((fea_train.shape[0], lbls.max().astype(int)))

    index_seg_all = []
    for j in range(len_seg_lbls):
        index_seg = np.where(seg_labels == unique_seg_labels[j])
        index_seg = np.column_stack(index_seg)
        bb = len(index_seg)
        cc = np.sum(fts[index_seg, :], axis=0) / bb
        mean_pixel = np.concatenate((mean_pixel, cc))
    mean_pixel = np.delete(mean_pixel, 0, axis=0)
    dist1 = compute_dist(np.array(fts), np.array(mean_pixel))
    dist_mean1 = compute_dist(np.array(mean_pixel), np.array(mean_pixel))
    spatial_coordinates = sptial_neighbor_matrix(index_all[0], gt)  #
    mean_coordinates = np.zeros((1, spatial_coordinates.shape[1]))  # max_seg_lab
    for jj in range(len_seg_lbls):
        index_seg = np.where(seg_labels == unique_seg_labels[jj])  #
        index_seg = np.column_stack(index_seg)
        bb = len(index_seg)
        cc = np.sum(spatial_coordinates[index_seg, :], axis=0) / bb
        mean_coordinates = np.concatenate((mean_coordinates, cc))
    mean_coordinates = np.delete(mean_coordinates, 0, axis=0)  #
    dist3 = compute_dist(np.array(spatial_coordinates), np.array(mean_coordinates))
    dist3 = dist3 / np.tile(np.sqrt(np.sum(dist3 ** 2, 1))[..., np.newaxis], (1, dist3.shape[1]))
    dist_mean3 = compute_dist(np.array(mean_coordinates), np.array(mean_coordinates))
    dist_mean3 = dist_mean3 / np.tile(np.sqrt(np.sum(dist_mean3 ** 2, 1))[..., np.newaxis], (1, dist_mean3.shape[1]))

    dist = dist1 + 50 * dist3  # 30  +parameter[r]
    lam = 15  # parameter[r1]  #indian 10  pavia 5
    S = np.zeros((np.array(fts).shape[0], len_seg_lbls))
    for k in range(np.array(fts).shape[0]):
        idxa0 = range(len_seg_lbls)
        di = dist[k, :]
        ad = -0.5 * lam * di
        S[k][idxa0] = EProjSimplex_new(ad)

    dist_mean = dist_mean1 + 50 * dist_mean3
    beta = 15  # indian :12 pavia 5 lam1
    S_mean = np.zeros((len_seg_lbls, len_seg_lbls))
    for k in range(len_seg_lbls):
        idxa0 = range(len_seg_lbls)
        di = dist_mean[k, :]
        ad = -0.5 * beta * di
        S_mean[k][idxa0] = EProjSimplex_new(ad)

    S_mean = S_mean - np.diag(np.diag(S_mean))
    S_mean = (S_mean + S_mean.transpose()) / 2
    # DS = np.sum(S_mean, axis=1)
    DS = np.mat(np.diag(np.sum(S_mean, axis=1)))
    L_S_mean = DS - S_mean
    S_mean_opt = (np.eye(len_seg_lbls) + L_S_mean / 0.1).I  # 0.1
    S_mean_opt = np.array(S_mean_opt)

    S = np.dot(S, S_mean_opt)
    eps = 1e-5
    DE = np.sum(S, axis=0)
    invDE = np.mat(np.diag(np.power(DE + eps, -1)))
    # pdb.set_trace() ######################################################################
    #########################################################################################

    #########################################################################################
    G = np.dot(S, invDE).dot(S.transpose())

    lbls = lbls
    # lbls = lbls.astype(np.long)
    idx = range(len(lbls))
    # idx_train = idx[0:len(idx_train_index)]
    # idx_test = idx[len(idx_train_index):]
    #
    # return fts, lbls, idx_train_index, idx_train, idx_test_index, idx_test, G, mean_pixel, index_all, ground_truth
    # return data, gt
    return fts, lbls, G, index_all[0]


def DataLoader1(dataset_name):
    data = []
    if (dataset_name == 'IP'):
        print("Indian Pine")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/IndianPine/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/IndianPine/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        # class_count = 17
        # train_ratio = 0.015
        # num_graph = 6

    if (dataset_name == 'PU'):
        print("PaviaU")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/paviau/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/paviau/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        # class_count = 10
        # train_ratio = 0.005
        # num_graph = 16

    if (dataset_name == 'HS'):
        print("Houston")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/houston/Houston.mat')
        data = data_mat['Houston']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/houston/Houston_gt.mat')
        gt = gt_mat['Houston_gt']
        # class_count = 16
        # train_ratio = 0.01
        # num_graph = 6

    if (dataset_name == 'SA'):
        print("Salinas")
        data_mat = sio.loadmat('/home/wjx/Cluster_Group/data/salinas/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('/home/wjx/Cluster_Group/data/salinas/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        # class_count = 17
        # train_ratio = 0.01
        # num_graph = 30

    gnd = np.array(gt)
    gnd = np.transpose(gnd)

    gnd = np.array(gnd)
    gt = gnd
    data_all = np.array(data)
    data_all = (data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all))  # 对数据进行标准化
    # img = io.imread("D:\yunding\paper_four\HSI_data\Indian_29_42_89.png")
    data_seg = slic(data_all, n_segments=2000, compactness=0.1)  # indian slic 分割 1500, 0.01; pavia 分割 1500, 1
    # out = mark_boundaries(img, data_seg)
    # plt.imshow(out)
    # plt.show()
    # sio.savemat('data_seg.mat', {'data_seg': data_seg})
    print('this is data shape', data_all.shape)

    row_gnd = np.size(gnd, 0)
    col_gnd = np.size(gnd, 1)
    dim = np.size(data_all, 2)
    gnd = np.reshape(gnd, (1, row_gnd * col_gnd))  # ground truth
    ground_truth = gnd.copy()
    fea = np.reshape(data_all, (1, row_gnd * col_gnd, dim))
    fea = np.squeeze(fea)
    # kmeans = KMeans(n_clusters=100, random_state=0).fit(fea)
    # data_seg = kmeans.labels_

    gnd = np.squeeze(gnd)
    ground_truth = gnd.copy()
    choice = np.where(gnd != 0)
    segments = np.reshape(data_seg, (1, row_gnd * col_gnd))
    segments = np.squeeze(segments)
    num_class = gnd.max()
    num_class = num_class.astype(int)
    index_all_block = []
    num_Per_class = np.zeros((1, num_class))

    """ idx_train idx_test """
    # training = [30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30] #indian %
    # training = [30,30,30,30,30,30,15,30,15,30,30,30,30,30,30,30]
    # training = [40, 40, 40, 40, 40, 40, 15, 40, 15, 40, 40, 40, 40, 40, 40, 40]  # indian
    # training = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    training = [3, 4, 2, 2, 3, 4, 4, 12, 7, 4, 2, 2, 1, 2, 8, 2]
    # training = [7, 19, 3, 4, 2, 6, 2, 4, 1]
    # training = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]

    training = np.array(training)
    per_class_train_num = np.zeros((1, num_class))
    idx_train_index = []
    idx_test_index = []
    for i in range(num_class):
        ii = i + 1.
        a2 = np.where(gnd == ii)
        a2 = np.array(a2)
        a2 = np.squeeze(a2)
        num_Per_class[0, np.array(ii - 1.).astype(int)] = len(a2)
        index_all_block = np.concatenate((index_all_block, a2))
        # idx_train_1 = random.sample(list(a2), np.floor(0.1 * len(a2)).astype(int))
        # idx_train_1 = random.sample(list(a2), per_class_sample)
        idx_train_1 = random.sample(list(a2), training[i])
        idx_train_index = np.concatenate((idx_train_index, idx_train_1))
        per_class_train_num[0, np.array(ii - 1.).astype(int)] = len(idx_train_index)
        # pdb.set_trace()
        idx_test_1 = np.setdiff1d(a2, idx_train_index, True)
        idx_test_index = np.concatenate((idx_test_index, idx_test_1[0:np.ceil(len(idx_test_1)).astype(int)]))

    index_all = np.concatenate((idx_train_index.astype(int), idx_test_index.astype(int)))
    fea_train = fea[idx_train_index.astype(int), :]
    fea_test = fea[idx_test_index.astype(int), :]
    fts = np.concatenate((fea_train, fea_test))  # fts = samples*dimension
    lbls = np.concatenate((gnd[idx_train_index.astype(int)], gnd[idx_test_index.astype(int)]))  # find labels
    seg_labels = np.concatenate((segments[idx_train_index.astype(int)], segments[idx_test_index.astype(int)]))
    unique_seg_lbls = np.unique(seg_labels)
    len_seg_lbls = len(unique_seg_lbls)  # find the unique length of segment labels
    mean_pixel = np.zeros((1, dim))
    unique_seg_labels = np.unique(seg_labels)
    max_seg_lab = np.max(segments)

    YY = np.zeros((fea_train.shape[0], lbls.max().astype(int)))

    index_seg_all = []
    for j in range(len_seg_lbls):
        index_seg = np.where(seg_labels == unique_seg_labels[j])
        index_seg = np.column_stack(index_seg)
        bb = len(index_seg)
        cc = np.sum(fts[index_seg, :], axis=0) / bb
        mean_pixel = np.concatenate((mean_pixel, cc))
    mean_pixel = np.delete(mean_pixel, 0, axis=0)
    dist1 = compute_dist(np.array(fts), np.array(mean_pixel))
    dist_mean1 = compute_dist(np.array(mean_pixel), np.array(mean_pixel))
    spatial_coordinates = sptial_neighbor_matrix(index_all, gt)  #
    mean_coordinates = np.zeros((1, spatial_coordinates.shape[1]))  # max_seg_lab
    for jj in range(len_seg_lbls):
        index_seg = np.where(seg_labels == unique_seg_labels[jj])  #
        index_seg = np.column_stack(index_seg)
        bb = len(index_seg)
        cc = np.sum(spatial_coordinates[index_seg, :], axis=0) / bb
        mean_coordinates = np.concatenate((mean_coordinates, cc))
    mean_coordinates = np.delete(mean_coordinates, 0, axis=0)  #
    dist3 = compute_dist(np.array(spatial_coordinates), np.array(mean_coordinates))
    dist3 = dist3 / np.tile(np.sqrt(np.sum(dist3 ** 2, 1))[..., np.newaxis], (1, dist3.shape[1]))
    dist_mean3 = compute_dist(np.array(mean_coordinates), np.array(mean_coordinates))
    dist_mean3 = dist_mean3 / np.tile(np.sqrt(np.sum(dist_mean3 ** 2, 1))[..., np.newaxis], (1, dist_mean3.shape[1]))

    dist = dist1 + 50 * dist3  # 30  +parameter[r]
    lam = 15  # parameter[r1]  #indian 10  pavia 5
    S = np.zeros((np.array(fts).shape[0], len_seg_lbls))
    for k in range(np.array(fts).shape[0]):
        idxa0 = range(len_seg_lbls)
        di = dist[k, :]
        ad = -0.5 * lam * di
        S[k][idxa0] = EProjSimplex_new(ad)

    dist_mean = dist_mean1 + 50 * dist_mean3
    beta = 15  # indian :12 pavia 5 lam1
    S_mean = np.zeros((len_seg_lbls, len_seg_lbls))
    for k in range(len_seg_lbls):
        idxa0 = range(len_seg_lbls)
        di = dist_mean[k, :]
        ad = -0.5 * beta * di
        S_mean[k][idxa0] = EProjSimplex_new(ad)

    S_mean = S_mean - np.diag(np.diag(S_mean))
    S_mean = (S_mean + S_mean.transpose()) / 2
    # DS = np.sum(S_mean, axis=1)
    DS = np.mat(np.diag(np.sum(S_mean, axis=1)))
    L_S_mean = DS - S_mean
    S_mean_opt = (np.eye(len_seg_lbls) + L_S_mean / 0.1).I  # 0.1
    S_mean_opt = np.array(S_mean_opt)

    S = np.dot(S, S_mean_opt)
    eps = 1e-5
    DE = np.sum(S, axis=0)
    invDE = np.mat(np.diag(np.power(DE + eps, -1)))
    # pdb.set_trace() ######################################################################
    #########################################################################################

    #########################################################################################
    G = np.dot(S, invDE).dot(S.transpose())

    lbls = lbls - 1
    # lbls = lbls.astype(np.long)
    idx = range(len(lbls))
    idx_train = idx[0:len(idx_train_index)]
    idx_test = idx[len(idx_train_index):]
    #
    # return fts, lbls, idx_train_index, idx_train, idx_test_index, idx_test, G, mean_pixel, index_all, ground_truth
    # return data, gt
    return fts, lbls, G, index_all

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
    data = minMax.fit_transform(
        data)  # 将数据转换为标准正态分布（均值为0,标准差为1）。通过从每个特征中减去样本均值并除以标准差来实现的。它常用于那些假定输入特征为正态分布的模型，如许多线性模型和神经网络。
    # data = np.reshape(data, [height, width, bands])
    return data  # 145*145, 60


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

