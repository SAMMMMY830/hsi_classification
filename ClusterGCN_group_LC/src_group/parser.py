import argparse


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run.")

    # Indina_Pine
    # parser.add_argument("--edge-path",
    #                     nargs = "?",
    #                     default = "/data/lc/train/Cluster-GCN/ClusterGCN-master/input/egde_indian2.csv",
    #                 help = "Edge list csv.")
    # parser.add_argument("--Indian-Target",
    #                     nargs="?",
    #                     default="/data/lc/train/Cluster-GCN/ClusterGCN-master/input/Indian_pines_gt.mat",
    #                     help="Edge list csv.")
    # parser.add_argument("--features-path",
    #                     nargs = "?",
    #                     default = "/data/lc/train/Cluster-GCN/ClusterGCN-master/input/features_indian2.csv",
    #                 help = "Features json.")
    #
    # parser.add_argument("--target-path",
    #                     nargs = "?",
    #                     default = "/data/lc/train/Cluster-GCN/ClusterGCN-master/input/target_indian2.csv",
    #                 help = "Target classes csv.")
    #
    parser.add_argument("--save-path",
                        nargs="?",
                        default="/home/wjx/Cluster_Group/output/",
                        help="Target classes csv.")

    # PaviaU
    # parser.add_argument("--edge-path",
    #                     nargs="?",
    #                     default="/home/lc/data/CG/input/egde_PaviaU_all.csv",
    #                     help="Edge list csv.")
    # parser.add_argument("--PaviaU-Target",
    #                     nargs="?",
    #                     default="/home/lc/data/CG/input/PaviaU_sub_gt.mat",
    #                     help="Edge list csv.")
    # parser.add_argument("--features-path",
    #                     nargs="?",
    #                     default="/home/lc/data/CG/input/features_PaviaU_all.csv",
    #                     help="Features json.")
    #
    # parser.add_argument("--target-path",
    #                     nargs="?",
    #                     default="/home/lc/data/CG/input/target_PaviaU_all.csv",
    #                     help="Target classes csv.")

    # PaviaU
    # parser.add_argument("--edge-path",
    #                     nargs="?",
    #                     default="/data/lc/train/Cluster-GCN/ClusterGCN-master/input/egde_Houston1.csv",
    #                     help="Edge list csv.")
    # parser.add_argument("--Houston-Target",
    #                     nargs="?",
    #                     default="/data/lc/train/Cluster-GCN/ClusterGCN-master/input/Houston_gt.mat",
    #                     help="Edge list csv.")
    # parser.add_argument("--features-path",
    #                     nargs="?",
    #                     default="/data/lc/train/Cluster-GCN/ClusterGCN-master/input/features_Houston1.csv",
    #                     help="Features json.")
    #
    # parser.add_argument("--target-path",
    #                     nargs="?",
    #                     default="/data/lc/train/Cluster-GCN/ClusterGCN-master/input/target_Houston1.csv",
    #                     help="Target classes csv.")

    parser.add_argument("--clustering-method",
                        nargs="?",
                        default="generate",
                        help="Clustering method for graph decomposition. Default is the metis procedure.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Number of training epochs. Default is 200.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--cuda",
                        type=int,
                        default=1,
                        help="Random seed for train-test split. Default is 42.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.01,
                        help="Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.03,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.99,
                        help="Test data ratio. Default is 0.1.")
    parser.add_argument("--train_num",
                        type=float,
                        default=13,
                        help="Test data ratio. Default is 0.1.")

    parser.add_argument("--cluster-number",
                        type=int,
                        default=6,
                        help="Number of clusters extracted. Default is 10.")
    # model
    parser.add_argument('--num-layers', type=int, default=10, help='Number of GNN layers.')
    parser.add_argument('--hid-dim', type=int, default=60, help='Hidden channel size.')
    parser.set_defaults(layers=[16, 16, 16])
    # learnable parameters in aggr
    parser.add_argument('--min_num', type=float, default=0.25)
    parser.add_argument('--graph_skip_conn', type=float, default=0.15)
    parser.add_argument('--update_adj_ratio', type=float, default=0.15)
    parser.add_argument('--feat_adj_dropout', type=float, default=0.01)
    parser.add_argument('--balanced_degree', type=float, default=0.6)
    parser.add_argument('--epsilon', type=float, default=0.55, help='Value of epsilon in changing adj.')
    parser.add_argument('--metric_type', nargs="?", default="weighted_cosine", help='Value of epsilon in changing adj.')
    parser.add_argument('--degree_ratio', type=float, default=0.1)
    parser.add_argument('--sparsity_ratio', type=float, default=0.1)
    parser.add_argument('--smoothness_ratio', type=float, default=0.5)

    parser.add_argument("--Dataset_Name", nargs="?", default="IP", help="Target classes csv.")

    # add group
    parser.add_argument('--num_group', nargs='*', default=[4], type=int, help='reduction ratio of key')
    parser.add_argument('--att_norm', type=str, default='softmax')
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--loss_emb', type=str, default='IG_binomial_deviance')
    parser.add_argument('--loss_div', type=str, default='div_bd')
    parser.add_argument('--bias', nargs='*', default=True, type=bool)
    parser.add_argument('--weight_decay', nargs='*', default=[0.00] * 5, type=float, help='conv, bn, que, key, val')
    parser.add_argument('--lam_glb', default=0.01, type=float)
    parser.add_argument('--lam_div', nargs='*', default=[0.7], type=float)
    parser.add_argument('--add_global_group', action='store_true')
    parser.add_argument('--top_k', default=11, type=int)
    parser.add_argument('--grouping-layer', type=str, default='attention')

    return parser.parse_args()
