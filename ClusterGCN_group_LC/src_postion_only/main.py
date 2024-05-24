import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from src.parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, feature_reader, DataLoader,graph_reader,label_reader
from position import PositionEncoding

def train_test(args, CM):

    epoch = 10
    loss_mean = 0.0
    OA_mean= 0.0
    Kap_mean= 0.0
    AA_mean = 0.0
    OA1 = []
    kap1 = []
    AA1 = []

    for i in range(epoch):
        # 初始化模型
        PE = PositionEncoding(args, CM.graph)   # 加载位置矩阵文件pca_matrix
        model = ClusterGCNTrainer(args, CM, PE)
        loss = model.train()
        OA, Kap, AA = model.test()
        OA1.append(OA)
        kap1.append(Kap)
        AA1.append(AA)
        loss_mean = float(loss) + loss_mean
        OA_mean = float(OA) + OA_mean
        Kap_mean = float(Kap) + Kap_mean
        AA_mean = float(AA) + AA_mean

    OA_mean /= epoch
    Kap_mean /= epoch
    AA_mean /= epoch
    loss_mean /= epoch
    OA1_dict = np.append(OA1, OA_mean)
    AA1_dict = np.append(AA1, AA_mean)
    kap1_dict = np.append(kap1, Kap_mean)
    # 创建字典
    result_dict = {
        'OA': OA1_dict,
        'Kap': kap1_dict,
        'AA': AA1_dict,
    }

    # 将字典转换为DataFrame
    result_df = pd.DataFrame(result_dict)

    # 保存为CSV文件
    timestamp = time.time()
    save_path = args.save_path
    save_file_path = save_path + '/test'+str(timestamp)+'.csv'
    result_df.to_csv(save_file_path, index=False)
    print("\nOA: ", OA1,
          "\nKap:", kap1,
          "\nAA:", AA1,
              )
    print("\nOA_mean: {:.4f}".format(OA_mean),
          "Kap_mean:{:.4f}".format(Kap_mean),
          "AA_mean:{:.4f}".format(AA_mean),
          "Train_loss:{:.4f}".format(loss_mean))


def main(args):

    # 设置随机数种子，保证实验结果的可再现
    torch.manual_seed(args.seed)
    class_count, train_ratio, num_graph = DataLoader(args.Dataset_Name)
    G_path = os.path.join(args.graph_path, args.Dataset_Name,"edge_list.csv")
    G_fea_path = os.path.join(args.graph_path, args.Dataset_Name,"features.csv")
    G_gt_path = os.path.join(args.graph_path, args.Dataset_Name,"gt.csv")

    # 读文件获取图、特征、标签
    G = graph_reader(G_path)
    G_features = feature_reader(G_fea_path)
    G_label = label_reader(G_gt_path)
    CM = ClusteringMachine(args, G, G_features, G_label, class_count, train_ratio, num_graph)

    # 拆分子图，返回self.sg_features, self.sg_targets, self.sg_nodes
    CM.decompose()
    # 训练，测试
    train_test(args, CM)


if __name__ == "__main__":
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # 配置参数
    args = parameter_parser()
    args.embedding_dim = [60 * i for i in args.num_group]
    main(args)
