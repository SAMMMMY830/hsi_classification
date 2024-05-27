import pickle
from Outils import losses
import torch
import random
import numpy as np
from tqdm import trange, tqdm
from layers import StackedGCN
import copy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from src.deep_models3 import DeeperGCN
from utils import *
from clustering import train_test_spilts
class ClusterGCNTrainer(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, args, CM, PE):
        """
        :param ags: Arguments object.
        :param gt: 10249
        :param gnd: 145*145
        :param g_label: 21025
        features: 10249

        """
        self.args = args
        # 初始化layer矩阵，0，cluster_number*cluster_number
        self.CM = CM
        # self.nodes = self.CM.nodes
        self.num_graph = self.CM.num_graph
        # self.layer = np.zeros(self.num_graph)
        # self.best_layer = np.zeros(self.num_graph)
        self.index = self.CM.node_ids  # 所有节点
        self.gt = self.CM.gt  # 10249
        self.node_id_rl = self.CM.node_id_rl
        self.features = self.CM.features  # 所有节点特征 10249
        self.orig_feature = self.CM.orig_feature
        self.gnd = self.CM.gnd   # 145* 145
        self.g_label = self.CM.g_label  # 21025
        self.clusters = self.CM.clusters
        self.sub_feat = self.CM.sg_features  # 子图特征列表
        self.sg_targets = self.CM.sg_targets  # 子图标签列表
        self.sg_nodes = self.CM.sg_nodes  # 子图节点列表
        self.train_ratio = self.CM.train_ratio  # 训练集与测试集分割比
        self.class_count = self.CM.class_count  # 分类数
        self.args.device = self.device = torch.device(self.args.cuda if torch.cuda.is_available() else "cpu")
        self.PE = PE
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        self.train_node, self.test_nodes, self.total_sample_num, self.total_test_num = train_test_spilts(self.args, self.class_count, self.clusters, self.sub_feat, self.sg_targets, self.sg_nodes, self.train_ratio)
        adj_file = 'adj.pkl'
        self.degree = {}
        self.adj = {}
        for i in self.clusters:
            self.adj[i] = self.edge_construction(self.train_node[i], self.sg_nodes[i],self.train_node[i])  # train_node索引，sg_nodes子图节点
            self.degree[i] = self.adj[i].sum(0)

        # 保存 self.adj 到文件
        with open(adj_file, 'wb') as file:
            pickle.dump(self.adj, file)
        self.model = DeeperGCN(
            args=self.args,
            node_id_rl = self.node_id_rl,
            adj=self.adj,
            node_feat_dim=self.features.shape[1],
            hid_dim=self.args.hid_dim,
            out_dim=self.class_count,
            clusters=self.clusters,
            degree_max=self.sg_nodes,
            PE=self.PE
        )
        self.model = self.model.to(self.device)

        # if os.path.exists(adj_file):
        #     with open(adj_file, 'rb') as file:
        #         self.adj = pickle.load(file)
        #         for i in range(self.clusters):
        #             self.degree[i] = self.adj[i].sum(0)

        self.model_layer = []

    def do_forward_pass(self, cluster, Cost_emb, Cost_div):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        train_nodes = self.train_node[cluster].to(self.device)  # 序号
        features = self.CM.sg_features[cluster].to(self.device)     #
        target = self.CM.sg_targets[cluster].to(self.device).squeeze()
        degree = self.degree[cluster].to(self.device)
        adj = self.adj[cluster].to(self.device)
        sg_nodes = self.sg_nodes[cluster]
        # 将子图的边和特征传入，得到预测值，并计算损失
        # layer_dyn = np.zeros(self.args.cluster_number)

        x_group, x_n, entropy_loss = self.model(sg_nodes, features, train_nodes, cluster, 'train', degree, adj, target)
        # self.layer[cluster] = layer
        loss_emb = Cost_emb(x_group[train_nodes], x_n[train_nodes], train_nodes)  # InfoGraphBDLoss 空间内损失
        loss_div = Cost_div(x_group[train_nodes], x_group[train_nodes])  # DivBD 空间间损失
        contrastive_loss = loss_emb + self.args.lam_div[0] * loss_div
        # self.layer[cluster] = layer
        # self.best_layer[cluster] = self.layer[cluster] + self.best_layer[cluster]
        node_count = train_nodes.shape[0]
        average_loss = entropy_loss*self.args.alpha + contrastive_loss*(1-self.args.alpha)
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster.
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        sg_nodes = self.sg_nodes[cluster]
        test_nodes = self.test_nodes[cluster].to(self.device)
        features = self.CM.sg_features[cluster].to(self.device)
        target = self.CM.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        degree = self.degree[cluster].to(self.device)
        # index_graph = self.CM.index_graph[cluster]
        adj = self.adj[cluster].to(self.device)
        # index_all = self.CM.sg_nodes[cluster].to(self.device)

        # 取出测试节点的预测值，size为（n,3）
        prediction = self.model(sg_nodes, features, test_nodes, cluster, "test", degree, adj, target)
        # prediction = self.model(edges, features)
        prediction = prediction[test_nodes, :]
        return prediction, target

    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def train(self):
        """
        Training a model.
        """
        print("Training started.\n")
        # 展示进度条
        epochs = trange(self.args.epochs, desc = "Train Loss")
        # 选择损失函数
        to_optim = set_lr_wd(self.model, self.args)
        Cost_emb, to_optim = losses.select_loss(self.args.loss_emb, self.args, self.device,to_optim)  # IG_binomial_deviance = InfoGraphBDLoss
        Cost_div, to_optim = losses.select_loss(self.args.loss_div, self.args, self.device, to_optim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # 开启训练模式
        self.model.train()
        loss = []
        for epoch in epochs:
            # 打乱子图顺序
            loss_all = 0
            random.shuffle(self.CM.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.CM.clusters:
                self.optimizer.zero_grad()
                loss_sum, node_count = self.do_forward_pass(cluster, Cost_emb, Cost_div)
                loss_sum.backward()
                self.optimizer.step()
                loss_all += loss_sum * node_count
                # average_loss = self.update_average_loss(loss_sum, node_count)
            # print("layer",self.layer)
            average_loss = loss_all / len(self.index)
            # print('Epoch {}, Loss {:.3f}, Loss_club {:.3f}'.format(epoch, loss_all / len(self.CM.index), loss_club / len(self.CM.index)))
            epochs.set_description("Train Loss: %g" % round(average_loss.item(), 4))

        return average_loss
        # fig, ax = plt.subplots()
        # ax.plot(loss, label='训练损失')
        # # 设置图形属性
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # ax.set_title('损失变化曲线')
        # ax.legend()
        # plt.show()

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        # 进入评估模式
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.CM.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions)
        # self.predictions = np.concatenate(self.predictions).argmax(1)
        OA, Kap, AA= Cal_accuracy(self.predictions, self.targets)
        print("\nOA: {:.4f}".format(OA),
              "Kap:{:.4f}".format(Kap),
              "AA:{:.4f}".format(AA)
              )
        return OA , Kap, AA

    def edge_construction(self, train_node, node_index, train_node_rl):  # train_node 训练集节点；node_index：子图节点；
                        # (self.train_node[i], self.sg_nodes[i], self.train_node[i])
        """Load citation network dataset (cora only for now)"""
        # print('Loading {} dataset...'.format(dataset))  path="../data/cora/", dataset="cora",
        ran_num_sample = 2  # Indian_pines_PCA  PaviaU_sub Houston_PCA
        g_label = self.g_label[node_index]  # 根据子图中的节点选取对应标签
        # gt = np.squeeze(gnd)    # 去掉长度为1的维度
        features = self.orig_feature[node_index]    # 选取节点特征

        print('this is features', np.array(features).shape)
        # dist1 可以理解为光谱距离
        dist1 = compute_dist(self.args, np.array(features), np.array(features))
        # 将dist1中的所有值减去当前行的最大值。
        dist1 = torch.exp(dist1 - torch.max(dist1, dim=0, keepdim=True).values.expand_as(dist1))
        # dist1 = torch.exp(dist1 - torch.max(dist1, dim=0, keepdim=True).values.unsqueeze(0)).expand_as(dist1)
        # dist1 = np.exp(dist1 - np.tile(np.max(dist1, axis=0)[..., np.newaxis], np.size(dist1, 1)))
        spatial_corrdinates = sptial_neighbor_matrix(np.array(node_index), 3, self.gnd)
        # dist2 可以理解为空间坐标的距离
        dist2 = compute_dist(self.args, np.array(spatial_corrdinates), np.array(spatial_corrdinates))
        # dist2 = dist2 / np.tile(torch.sqrt(torch.sum(dist2 ** 2, 1)), (dist2.shape[0], 1))
        dist2 = dist2 / torch.sqrt(torch.sum(dist2 ** 2, dim=1, keepdim=True)).expand_as(dist2)
        # 这里的β=30
        dist = dist1 + 20 * dist2  # dist = dist1 + 30*dist2
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
        for j in range(np.max(g_label).astype(int) - 1):
            j = j + 1
            sum_1 = labeled_dist[j * ran_num_sample:(j + 1) * ran_num_sample,
                    j * ran_num_sample:(j + 1) * ran_num_sample]
            sum_2 = sum_2 + np.sum(sum_1)
        sum_intra_all = sum_2 + first_block_intra_dist_sum
        integer_intra = ran_num_sample * ran_num_sample * np.max(g_label)
        average_sum = sum_intra_all / (integer_intra)
        # print('this is labels_no_zero', gt)
        # print('this is 类内平均', average_sum)
        average_inter = (np.sum(labeled_dist) - np.sum(sum_intra_all)) / (labeled_dist.shape[0] ** 2 - integer_intra)
        # print('this is 类间平均', average_inter)
        # print('这是类内类间初始差值（阈值）', average_inter - average_sum)
        # 这里是去除自己到自己的一个距离，再把大于类内均值的设为inf
        dist_new = dist_new.cpu() - np.diag(np.diag(dist_new.cpu()))   # 自身距离设置为0
        dist_new[dist_new > average_sum] = float('inf')
        S_dist = np.exp(-dist_new / 6)  # 6
        lam = 0.01  # 5
        # 设置邻接矩阵的表达形式
        S = np.zeros((dist.shape[0], dist.shape[1]))
        # 先找到每一行的非0元素位置，得到值ai，再把dist矩阵中的对应元素赋给di，拿ai-di得到ad，将概率值设为1.
        for k in range(len(g_label)):
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

def set_lr_wd(model, args):
    conv_param, bn_param = [], []
    que_wei, key_wei, val_wei = [], [], []
    for name, param in model.named_parameters():
        print(name, param.shape)
        if 'convs' in name:
            conv_param.append(param)
        elif 'bns' in name:
            bn_param.append(param)
        elif 'w_q' in name:
            que_wei.append(param)
        elif 'w_k' in name:
            key_wei.append(param)
        elif 'w_v' in name:
            val_wei.append(param)
    to_optim = [{'params': conv_param, 'weight_decay': args.weight_decay[0], 'lr': args.learning_rate},
                {'params': bn_param, 'weight_decay': args.weight_decay[1], 'lr': args.learning_rate},
                {'params': que_wei, 'weight_decay': args.weight_decay[2], 'lr': args.learning_rate},
                {'params': key_wei, 'weight_decay': args.weight_decay[3], 'lr': args.learning_rate},
                {'params': val_wei, 'weight_decay': args.weight_decay[4], 'lr': args.learning_rate}]
    return to_optim