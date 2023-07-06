# -*- coding: utf-8 -*-
# @Time    : 6/1/23 12:40 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : compute_mAP.py

# compute mAP on whisper-at

from stats import *

mdl_size_list = [
'tiny_low_False',
'tiny.en_low_False',
'base_low_False',
'base.en_low_False',
'small_low_False',
'small_low_True',
'small.en_low_False',
'small.en_low_True',
'medium_low_False',
'medium_low_True',
'medium.en_low_False',
'medium.en_low_True',
'large-v1_low_False',
'large-v1_low_True',
'large-v2_low_False',
'large-v2_low_True']

for mdl_size in mdl_size_list:
    all_truth = np.load('/data/sls/scratch/yuangong/whisper-at/old/at_res/all_truth_' + mdl_size + '.npy')
    all_pred = np.load('/data/sls/scratch/yuangong/whisper-at/old/at_res/all_pred_' + mdl_size + '.npy')
    print(mdl_size)
    print(all_truth.shape, all_pred.shape)

    stats = calculate_stats(all_pred, all_truth)
    mAP = np.mean([stat['AP'] for stat in stats])
    print(mAP)