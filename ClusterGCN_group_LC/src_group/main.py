import time

import numpy as np
import pandas as pd
import torch
from src_group.parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, feature_reader, DataLoader


def train_test(args, CM):

    epoch = 3
    loss_mean = 0.0
    OA_mean= 0.0
    Kap_mean= 0.0
    AA_mean = 0.0
    OA1 = []
    kap1 = []
    AA1 = []
    for i in range(epoch):
        # 初始化模型
        gcn_trainer = ClusterGCNTrainer(args, CM)
        loss = gcn_trainer.train()
        OA, Kap, AA = gcn_trainer.test()
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


def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # 配置参数
    args = parameter_parser()
    args.embedding_dim = [60 * i for i in args.num_group]
    # 设置随机数种子，保证实验结果的可再现
    torch.manual_seed(args.seed)
    # 可视化参数表制作
    tab_printer(args)
    # 读取边索引
    orig_features, orig_label, class_count, train_ratio, num_graph = DataLoader(args.Dataset_Name)
    features = feature_reader(orig_features)  # 光谱特征200→60 的包含0标签的特征
    # 对聚类器做一个初始化
    CM = ClusteringMachine(args, features, orig_label, class_count, train_ratio, num_graph)
    # 拆分子图
    CM.decompose()
    # 训练，测试
    train_test(args, CM)


if __name__ == "__main__":
    main()
