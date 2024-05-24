import argparse


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run.")
    #
    parser.add_argument("--save-path",
                        nargs="?",
                        default="/home/wjx/Cluster_Group/output/",
                        help="Target classes csv.")


    parser.add_argument("--clustering-method",
                        nargs="?",
                        default="generate",
                        help="Clustering method for graph decomposition. Default is the metis procedure.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train-test split. Default is 42.")

    parser.add_argument("--cuda",
                        type=int,
                        default=0,
                        help="device id")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.01,
                        help="Dropout parameter. Default is 0.5.")
    # model
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

    # PADEL
    parser.add_argument("--position_dim", nargs='*', default=60, type=int, help="whether using pca")
    parser.add_argument("--pca_dim", nargs='*', default=60, type=int, help="whether using pca")
    parser.add_argument("--graph-path", nargs="?", default="/home/wjx/Cluster_Group/input/", help="dataset name")
    parser.add_argument('--aggregator', type=str, default='mean', help='hidden_last size.')
    parser.add_argument('--pca', type=bool, default=True, help='whether using pca.')

    # lr,test_ratio,epochs
    parser.add_argument("--Dataset_Name", nargs="?", default="IP", help="dataset name")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.99,
                        help="Test data ratio. Default is 0.1.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Number of training epochs. Default is 200.")

    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers.')
    parser.add_argument('--hid-dim', type=int, default=60, help='Hidden channel size.')

    return parser.parse_args()
