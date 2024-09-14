import torch
import numpy as np

import argparse
import os, sys
import json
import math
import numpy as np
import copy
import time 

from config import config
from model import GCNext as Model
from datasets.h36m import H36MDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from datasets.h36m_eval import H36MEval
from test import test
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='baseline.txt', help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=48, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

parser.add_argument('--exp', type=str, default='baseline', help='=exp name')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate in first 30000 iters')
parser.add_argument('--lrr', type=float, default=1e-5, help='learning rate after 30000 iters')
parser.add_argument('--bs', type=int, default=256, help='=batch_size')
parser.add_argument('--worker', type=int, default=0, help='=num_worker')
parser.add_argument('--iters', type=int, default=40000, help='=number of iterations')
parser.add_argument('--c_iters', type=int, default=30000, help='=number of iterations')

parser.add_argument('--aux_lr', type=float, default=1e-8, help='learning rate for auxiliary params')    # 默认1e-7

parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48])
parser.add_argument('--tau', type=float, default=5, help='initial tau')    # 默认1e-7

args = parser.parse_args()


torch.use_deterministic_algorithms(True)
ensure_dir(f"ckpt/{args.exp}")
acc_log = open(f"ckpt/{args.exp}/test.txt", 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter(f"ckpt/{args.exp}/runs/{args.exp}")

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
args.with_normalization = True
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

config.batch_size = args.bs
config.num_workers = args.worker
config.cos_lr_total_iters = args.iters

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > args.c_iters:
        current_lr = args.lrr
    else:
        current_lr = args.lr
    optimizer.param_groups[0]["lr"] = current_lr
    optimizer.param_groups[1]["lr"] = args.aux_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr, tau) :

    if config.deriv_input:
        b,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :config.motion.h36m_input_length, :], h36m_motion_input_.cuda())
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred = model(h36m_motion_input_.cuda(), tau)
    motion_pred = torch.matmul(idct_m[:, :, :config.motion.h36m_input_length], motion_pred)

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    b,n,c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b,n,22,3).reshape(-1,3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b,n,22,3).reshape(-1,3)
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

model = Model(config, args.dyna)
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
model.train()
model.cuda()

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = H36MDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)
print(len(dataloader))

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = H36MEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=0, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
all_params = model.parameters()
aux_params = []
for pname, p in model.named_parameters():
    if any([pname.endswith(k) for k in ['adj_j', 'in_weight', 'out_weight', 'adj_t', 'adj_c', 'adj_tj', 'adj_jc', 'adj_tc']]):
        aux_params += [p]
params_id = list(map(id, aux_params))
other_params = list(filter(lambda p: id(p) not in params_id, all_params))


# initialize optimizer
optimizer = torch.optim.Adam([{'params': other_params},
                              {'params': aux_params, 'lr': args.aux_lr}],
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

logger = get_logger(f"ckpt/{args.exp}/log.log", 'train')

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

st = time.time()
stt = time.time()



tau = args.tau
tau_decay = -0.045

while (nb_iter + 1) < config.cos_lr_total_iters:
    for (h36m_motion_input, h36m_motion_target) in dataloader:
        if (nb_iter + 1) < args.c_iters - 1001:
            save_every = 5000
        elif (nb_iter + 1) < args.c_iters + 5001:
            save_every = 1000
        else:
            save_every = 2000

        if (nb_iter + 1) % len(dataloader) == 0:
            tau = tau * np.exp(tau_decay)

        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min, tau)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % save_every ==  0 :
            torch.save(model.state_dict(), f'ckpt/{args.exp}/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(eval_config, model, eval_dataloader, tau)
            print(f"Training Progress: {nb_iter + 1}/{config.cos_lr_total_iters}")
            print(f"Lr: {current_lr}")
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()
        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
