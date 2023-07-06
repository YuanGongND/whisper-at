# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
os.environ['TRANSFORMERS_CACHE'] = './tr/'
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader_feat as dataloader
import numpy as np
from traintest import train, validate
from models import TLTR

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--model_size", type=str, default='medium.en', help="The model size")
parser.add_argument("--label_smooth", type=float, default=0.0, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--ftmode", type=str, default='last', help="pretrained model path")
parser.add_argument("--pretrain_epoch", type=int, default=0, help="number of pretrained epochs")
parser.add_argument("--head_lr", type=float, default=1.0, help="learning rate ratio between mlp/base")
args = parser.parse_args()

if args.dataset == 'esc':
    if args.model_size == 'hubert-xlarge-ls960-ft' or args.model_size == 'wav2vec2-large-robust-ft-swbd-300h':
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/' + args.model_size
    else:
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/whisper_' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/whisper_' + args.model_size
    shuffle = True
elif args.dataset == 'as-bal' or args.dataset == 'as-full':
    if args.model_size == 'hubert-xlarge-ls960-ft' or args.model_size == 'wav2vec2-large-robust-ft-swbd-300h':
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/' + args.model_size
    else:
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/whisper_' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_eval/whisper_' + args.model_size
    shuffle = True

audio_conf = {'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'label_smooth': args.label_smooth, 'tar_path': train_tar_path}
val_audio_conf = {'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'tar_path': eval_tar_path}

if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

def get_feat_shape(path, args):
    mdl_size = args.model_size
    n_rep_dim_dict = {'tiny.en': 384, 'tiny': 384, 'base.en': 512, 'base': 512, 'small.en': 768, 'small': 768, 'medium.en': 1024, 'medium': 1024, 'large-v1': 1280, 'large-v2': 1280, 'wav2vec2-large-robust-ft-swbd-300h': 1024, 'hubert-xlarge-ls960-ft': 1280}
    n_layer_dict = {'tiny.en': 4, 'tiny': 4, 'base.en': 6, 'base': 6, 'small.en': 12, 'small': 12, 'medium.en': 24, 'medium': 24, 'large-v1': 32, 'large-v2': 32, 'wav2vec2-large-robust-ft-swbd-300h': 24, 'hubert-xlarge-ls960-ft': 48}
    return n_layer_dict[mdl_size], n_rep_dim_dict[mdl_size]

if 'whisper-high' in args.model:
    mode = args.model.split('-')[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layer, rep_dim = get_feat_shape(train_tar_path, args)
    print(mode, args.model_size, n_layer, rep_dim)
    audio_model = TLTR(label_dim=args.n_class, n_layer=n_layer, rep_dim=rep_dim, mode=mode)
else:
    raise ValueError('model not supported')

# use data parallel
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

code_path = args.exp_dir + '/src/'
if os.path.exists(code_path) == False:
    os.mkdir(code_path)
copy_path = '/data/sls/scratch/yuangong/whisper-a/src/'
os.system('cp ' + copy_path + '/*.sh ' + code_path)
os.system('cp ' + copy_path + '/*.py ' + code_path)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)


def wa_model(exp_dir, start_epoch=16, end_epoch=30):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        if os.path.exists(exp_dir + '/models/audio_model.' + str(epoch) + '.pth') == True:
            sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
            for key in sdA:
                sdA[key] = sdA[key] + sdB[key]
            model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    torch.save(sdA, exp_dir + '/models/audio_model_wa.pth')
    return sdA

# do model weight averaging
if args.wa == True:
    sdA = wa_model(args.exp_dir, args.wa_start, args.wa_end)
    msg = audio_model.load_state_dict(sdA, strict=True)
    print(msg)
    audio_model.eval()
    stats, _ = validate(audio_model, val_loader, args)
    wa_res = np.mean([stat['AP'] for stat in stats])
    print('mAP of model with weights averaged from checkpoint {:d}-{:d} is {:.4f}'.format(args.wa_start, args.wa_end, wa_res))
    np.savetxt(args.exp_dir + '/wa_res.csv', [args.wa_start, args.wa_end, wa_res], delimiter=',')