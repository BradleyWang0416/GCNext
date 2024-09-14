import torch
from torch import nn
from einops.layers.torch import Rearrange
from skeleton import Skeleton


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class GCBlock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        if not use_spatial_fc:

            # define update step
            self.update = Temporal_FC(seq)

            # define adjacency and mask for aggregation
            skl = Skeleton(skl_type='h36m', joint_n=22).skeleton
            skl = torch.tensor(skl, dtype=torch.float32, requires_grad=False)
            bi_skl = torch.zeros(22, 22, requires_grad=False)
            bi_skl[skl != 0] = 1.
            self.skl_mask = bi_skl.cuda()

            self.adj_j = nn.Parameter(torch.eye(22, 22))

            self.traj_mask = (torch.tril(torch.ones(seq, seq, requires_grad=False), 1) * torch.triu(
                torch.ones(seq, seq, requires_grad=False), -1)).cuda()  # 三对角矩阵
            for j in range(seq):
                self.traj_mask[j, j] = 0.
            self.adj_t = nn.Parameter(torch.zeros(seq, seq))

            self.adj_jc = nn.Parameter(torch.zeros(22, 3, 3))

            self.adj_tj = nn.Parameter(torch.zeros(dim, seq, seq))


        else:
            self.update = Spatial_FC(dim)



        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.update.fc.weight, gain=1e-8)

        nn.init.constant_(self.update.fc.bias, 0)

    def forward(self, x, mlp, if_make_dynamic, tau):
        # 输入[bs,66,50]

        b, v, t = x.shape
        x1 = x.reshape(b, v//3, 3, t)
        skl_mask = self.skl_mask
        x1 = torch.einsum('vj,bjct->bvct', self.adj_j.mul(skl_mask), x1)
        x1 = x1.reshape(b, v, t)

        traj_mask = self.traj_mask
        x2 = torch.einsum('ft,bnt->bnf', self.adj_t.mul(traj_mask), x)

        x3 = x.reshape(b, v//3, 3, t)
        x3 = torch.einsum('jkc,bjct->bjkt', self.adj_jc, x3)
        x3 = x3.reshape(b, v, t)


        x4 = torch.einsum('nft,bnt->bnf', self.adj_tj.mul(traj_mask.unsqueeze(0)), x)



        prob = torch.einsum('bj,jk->bk', x.mean(1), mlp)  # [bs,50]->[bs,4]
        if if_make_dynamic:
            gate = nn.functional.gumbel_softmax(prob, tau=tau, hard=True)
        else:
            gate = torch.tensor([1., 0., 0., 0.]).unsqueeze(0).expand(x.shape[0], -1).cuda()


        x2 = x2.unsqueeze(1)    # [bs,1,66,50]
        x3 = x3.unsqueeze(1)
        x4 = x4.unsqueeze(1)
        x_opts = torch.cat([torch.zeros_like(x1).cuda().unsqueeze(1), x2, x3, x4], dim=1)   # [bs,4,66,50]



        x_ = torch.einsum('bj,bjvt->bvt', gate, x_opts)



        x_ = self.update(x1 + x_)
        x_ = self.norm0(x_)
        x = x + x_

        return x

class TransGraphConvolution(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        self.layers = nn.Sequential(*[
            GCBlock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])

    def forward(self, x, mlp, if_make_dynamic, tau):
        x = self.layers(x, mlp, if_make_dynamic, tau)
        return x

def build_dynamic_layers(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransGraphConvolution(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    #if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")


