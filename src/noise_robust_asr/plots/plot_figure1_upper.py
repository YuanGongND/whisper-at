# -*- coding: utf-8 -*-
# @Time    : 3/4/23 9:11 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plot_snr.py
import numpy as np
from matplotlib import pyplot as plt

mdl_list = ['Whisper-Large', 'Hubert-XLarge-FT', 'Wav2vec2-Large-Robust-FT', 'Hubert-Large-FT', 'Wav2vec2-Base-FT']

exp_name_list = ['whisper_large-v1', 'hubert_xlarge', 'w2v_large_robust', 'hubert_large', 'w2v_base']
snr_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

for i in range(len(exp_name_list)):
    exp_name = exp_name_list[i]
    cur_res = np.loadtxt('/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results_camera/{:s}.csv'.format(exp_name))
    cur_res = cur_res * 100
    print(exp_name, cur_res.shape)
    if i == 0:
        plt.plot(snr_list, cur_res, '-o', label=mdl_list[i], linewidth=2)
    elif i == 1:
        plt.plot(snr_list, cur_res, 'g-x', label=mdl_list[i], linewidth=2)
    elif i == 2:
        plt.plot(snr_list, cur_res, 'c-*', label=mdl_list[i], linewidth=2)
    elif i == 3:
        plt.plot(snr_list, cur_res, '-^', label=mdl_list[i], linewidth=2)
    elif i == 4:
        plt.plot(snr_list, cur_res, 'r-d', label=mdl_list[i], linewidth=2)

plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=14)
plt.ylabel('Word Error Rate (%)', fontsize=14)
plt.legend(fontsize=10)
plt.gca().invert_xaxis()
plt.grid()
figure = plt.gcf()
figure.set_size_inches(6, 2.5)
plt.savefig('./snr_plot_cr.pdf', dpi=300, bbox_inches='tight')
plt.close()