# -*- coding: utf-8 -*-
# @Time    : 3/7/23 1:13 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : check_flops.py

# check model size and flops
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from high_mdls import HighMDL, HighMDLPool, HighMDLLayer, HighMDLFormal
# from whisper.model import Whisper, ModelDimensions

def cnt_flops(model, input):
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))
    print(flops.total()/1e9)
    print(flops.by_operator())
    #print(flops.by_module())
    #print(flops.by_module_and_operator())

# # original whisper model
# checkpoint_path = '/data/sls/scratch/yuangong/whisper-a/src/{:s}.pt'.format('small.en')
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# dims = ModelDimensions(**checkpoint["dims"])
# print(dims)
# model = Whisper(dims, label_dim=527, cla='mlp_1')
# input = torch.rand([1, 80, 512*2])
# cnt_flops(model, input)


def get_feat_shape(mdl_size):
    n_rep_dim_dict = {'tiny.en': 384, 'tiny': 384, 'base.en': 512, 'base': 512, 'small.en': 768, 'small': 768, 'medium.en': 1024, 'medium': 1024, 'large-v1': 1280, 'large-v2': 1280, 'wav2vec2-large-robust-ft-swbd-300h': 1024, 'hubert-xlarge-ls960-ft': 1280}
    n_layer_dict = {'tiny.en': 4, 'tiny': 4, 'base.en': 6, 'base': 6, 'small.en': 12, 'small': 12, 'medium.en': 24, 'medium': 24, 'large-v1': 32, 'large-v2': 32, 'wav2vec2-large-robust-ft-swbd-300h': 24, 'hubert-xlarge-ls960-ft': 48}
    return n_layer_dict[mdl_size], n_rep_dim_dict[mdl_size]

model_name = 'whisper-high-lw_down_tr_768_1_8'
model_size = 'large-v1'
mode = model_name.split('-')[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_layer, rep_dim = get_feat_shape(model_size)
print(mode, model_size, n_layer, rep_dim)
model = HighMDLFormal(label_dim=527, n_layer=n_layer, rep_dim=rep_dim, mode=mode)

# for large-v1
cnt_flops(model, torch.rand([1, n_layer, 25, rep_dim]))