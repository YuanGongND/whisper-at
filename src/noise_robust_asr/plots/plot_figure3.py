# -*- coding: utf-8 -*-
# @Time    : 2/13/23 3:40 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plot_figure3.py

# hist of best layer for each sound class

import os
os.environ["XDG_CACHE_HOME"] = './'
import numpy as np
from matplotlib import pyplot as plt

label_list = np.loadtxt('/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/esc_class_labels_indices.csv', delimiter=',', dtype=str, skiprows=1, usecols=(2)).tolist()

all_res = []
mdl_size = 'whisper_large-v1'
for fold in range(1, 6):
    for lr in [0.001]:
        cur_res = np.loadtxt('/data/sls/scratch/yuangong/whisper-a/src/baseline_cla/baseline_res/esc_{:s}_fold{:d}_lr_{:.4f}.csv'.format(mdl_size, fold, lr), delimiter=',', usecols=list(range(6, 56)))
        cur_res = cur_res[1:-1, :].tolist()  # exclude the input and last avg layer
        all_res.append(cur_res)
all_res = np.array(all_res)  # [5, 50] , 5 folds, 50 classes
sound_res = np.mean(all_res, axis=0) * 100

best_layer_list = []
for i in range(sound_res.shape[1]):
    best_layer_list.append(np.argmax(sound_res[:, i] + 1))

print(best_layer_list)

plt.hist(best_layer_list, bins=16, histtype ='bar', rwidth=0.7)
plt.xlabel('Representation of Layer', fontsize=14)
plt.ylabel('# Classes', fontsize=14)
plt.xticks(range(1, 33, 2))
plt.grid()
figure = plt.gcf()
figure.set_size_inches(6, 2)
plt.savefig('./formal_plot/best_layer.pdf', dpi=300, bbox_inches='tight')
plt.close()