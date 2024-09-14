import copy

import torch
from torch import nn
from dynamic_layer import build_dynamic_layers

from einops.layers.torch import Rearrange

class GCNext(nn.Module):
    def __init__(self, config, dyna_idx=None):
        self.config = copy.deepcopy(config)
        super(GCNext, self).__init__()
        seq = self.config.motion_mlp.seq_len


        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.dynamic_layers = build_dynamic_layers(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)   # nn.Linear(66,66)
            self.in_weight = nn.Parameter(torch.eye(50, 50))
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)  # nn.Linear(66,66)
            self.out_weight = nn.Parameter(torch.eye(50, 50))

        self.reset_parameters()


        self.mlp = nn.Parameter(torch.empty(50, 4))
        nn.init.xavier_uniform_(self.mlp, gain=1e-8)
        self.dyna_idx = dyna_idx

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, tau=1):
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)      # [bs,50,66]--nn.Linear(66,66)-->[bs,50,66]
            motion_feats = self.arr0(motion_feats)      # [bs,66,50]
            motion_feats = torch.einsum('bvt,tj->bvj', motion_feats, self.in_weight)

        for i in range(len(self.dynamic_layers.layers)):
            if_make_dynamic = True if self.dyna_idx[0] <= i <= self.dyna_idx[1] else False
            motion_feats = self.dynamic_layers.layers[i](motion_feats, self.mlp, if_make_dynamic, tau)


        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)      # [bs,50,66]
            motion_feats = self.motion_fc_out(motion_feats)     # [bs,50,66]--nn.Linear(66,66)-->[bs,50,66]
            motion_feats = torch.einsum('btv,tj->bjv', motion_feats, self.out_weight)

        return motion_feats

