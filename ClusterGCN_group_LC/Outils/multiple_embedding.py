import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from termcolor import cprint

class QGrouping(torch.nn.Module):
    def __init__(self, args, in_emb_dim, key_dim=160):
        super(QGrouping, self).__init__()
        cprint(f'Initail a query grouping layer', 'green')
        self.args = args
        self.embedding_dim = self.args.embedding_dim[0]   # 4000
        self.k = self.args.num_group[0]   # 4

        self.key_dim = key_dim  # 160
        self.val_dim = self.embedding_dim // self.k     # 1000
        self.dim_per_group = self.embedding_dim // self.k   # 1000
        self.att_norm = self.args.att_norm   # softmax
        self.bias = self.args.bias

        if self.bias == True:
            self.w_k = torch.nn.Conv2d(in_emb_dim, key_dim, kernel_size=1).to(self.args.device)   #
            self.w_q = torch.nn.Conv2d(key_dim, self.k, 1).to(self.args.device)  #
            self.w_v = torch.nn.Conv2d(in_emb_dim, self.val_dim, kernel_size=1).to(self.args.device)    #
        else:
            self.w_k = torch.nn.Conv2d(in_emb_dim, key_dim, kernel_size=1, bias=False).to(self.args.device)
            self.w_q = torch.nn.Conv2d(key_dim, self.k, 1, bias=False).to(self.args.device)
            self.w_v = torch.nn.Conv2d(in_emb_dim, self.val_dim, kernel_size=1, bias=False).to(self.args.device)

    # self.grouping(torch.cat(xs[self.start:], dim=-1), cluster)
    def forward(self, x):
        # x.unsqueeze(dim=a)：进行维度扩充，在维度为a的位置进行扩充
        key = self.w_k(x.unsqueeze(2).unsqueeze(3))  # 100
        val = self.w_v(x.unsqueeze(2).unsqueeze(3))
        # x.squeeze(dim)：进行维度压缩，去掉tensor中维数为1的维度
        val = val.squeeze()
        # reshape：不更改数据的情况下为数组赋予新形状，reshape(-1,4)：转为4列
        weights = self.w_q(key).reshape((-1, self.k))
        norm_w = []
        embs = []
        for i in range(x.size(0)):
            if self.att_norm == 'softmax':
                this_w = F.softmax(weights[i, :], dim=0).unsqueeze(0)
            elif self.att_norm == 'sigmoid':
                a = torch.sigmoid(weights[i, :])
                num_nodes = sum(x.size(0))
                this_w = a / torch.sqrt(num_nodes.float())
            else:
                raise ValueError
            this_val = val[i,:].unsqueeze(0)
            this_embs = torch.matmul(this_w.T, this_val)
            embs.append(this_embs.unsqueeze(0))
            norm_w.append(this_w)
        # torch.cat用于沿指定的维度拼接多个张量
        x_group = torch.cat(embs, dim=0)
        return x_group, norm_w

class MLGrouping(torch.nn.Module):
    def __init__(self, args, in_emb_dim) -> None:
        super(MLGrouping, self).__init__()
        cprint(f'Initail a mul_linear grouping layer', 'green')
        self.pool = args.pool
        self.k = args.num_group[0]
        # self.dim_per_group = per_group_dim
        self.ML_ops = torch.nn.ModuleList()
        for i in range(self.k):
            self.ML_ops.append(torch.nn.Linear(in_emb_dim, in_emb_dim))
        
    def forward(self, x):
        out = []
        for i in range(self.k):
            fea = self.ML_ops[i](x)   # torch.nn.Linear(in_emb_dim, self.dim_per_group) 进行k次线性层
            # if self.pool == 'mean':
            #     # fea = global_mean_pool(fea, batch=None)
            #     fea = fea.mean(dim=1)
            # else:
            #     # fea = global_add_pool(fea, batch=None)
            #     fea = fea.max(dim=1)
            out.append(fea.unsqueeze(1))
        x_group = torch.cat(out, dim=1)
        return x_group

