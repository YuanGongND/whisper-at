# -*- coding: utf-8 -*-
# @Time    : 2/13/23 3:40 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_summary.py

# summarize the results of esc experiments

import json
import os
os.environ["XDG_CACHE_HOME"] = './'
import numpy as np
from matplotlib import pyplot as plt

mdl_size_list =  ['whisper_large-v1',
                  'hubert-xlarge-ll60k',
                  'hubert-xlarge-ls960-ft',
                  'wav2vec2-large-robust',
                  'wav2vec2-large-robust-ft-swbd-300h',
                  'hubert-large-ls960-ft',
                  'wav2vec2-base-960h']

legend_list = ['Whisper-Large', 'Hubert-XLarge-PR', 'Hubert-XLarge-FT', 'Wav2vec2-Large-Robust-PR', 'Wav2vec2-Large-Robust-FT', 'Hubert-Large-FT', 'Wav2vec2-Base-FT']

for i, mdl_size in enumerate(mdl_size_list):
    all_res = []
    for fold in range(1, 6):
        for lr in [0.001]:
            cur_res = np.loadtxt('./baseline_res/esc_{:s}_fold{:d}_lr_{:.4f}.csv'.format(mdl_size, fold, lr), delimiter=',', usecols=(5)).tolist()
            all_res.append(cur_res)
    all_res = np.array(all_res)
    all_res = np.mean(all_res, axis=0)[1:-1] * 100
    print(all_res.shape)
    num_layer = all_res.shape[0]
    if i == 0: # whisper
        plt.plot(list(range(1, num_layer+1)), all_res, '-o', label = legend_list[i], linewidth=2)
    elif i == 1:
        plt.plot(list(range(1, num_layer + 1)), all_res, 'g-', label=legend_list[i], linewidth=2, alpha=0.5)
    elif i == 2:
        plt.plot(list(range(1, num_layer + 1)), all_res, 'g-x', label=legend_list[i], linewidth=2)
    elif i == 3:
        plt.plot(list(range(1, num_layer + 1)), all_res, 'c-', label=legend_list[i], linewidth=2, alpha=0.5)
    elif i == 4:
        plt.plot(list(range(1, num_layer + 1)), all_res, 'c-*', label=legend_list[i], linewidth=2)
    elif i == 5:
        plt.plot(list(range(1, num_layer + 1)), all_res, '-^', label=legend_list[i], linewidth=2)
    elif i == 6:
        plt.plot(list(range(1, num_layer + 1)), all_res, 'r-d', label=legend_list[i], linewidth=2)

plt.ylim([0, 1])
plt.xlabel('Classifying Using Representation of Layer # as Input', fontsize=13.5)
plt.ylabel('Sound Classification Accuracy (%)', fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.ylim([28, 90])
plt.xlim([0, 50])
figure = plt.gcf()
figure.set_size_inches(6, 4)
plt.savefig('./formal_plot/result_summary_' + str(lr) + '_cr.pdf', dpi=300, bbox_inches='tight')
plt.close()