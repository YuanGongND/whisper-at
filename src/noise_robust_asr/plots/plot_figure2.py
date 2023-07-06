# -*- coding: utf-8 -*-
# @Time    : 3/4/23 9:11 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plot_snr.py
import numpy as np
from matplotlib import pyplot as plt

label_list = np.loadtxt('/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/esc_class_labels_indices.csv', delimiter=',', dtype=str, skiprows=1, usecols=(2)).tolist()
print(label_list)

# classes not used in fitting the line
outlier_list = [8, 10, 11, 16, 22, 27, 35, 36, 41, 45, 46, 49]
not_outliear_list = []
for x in range(50):
    if x not in outlier_list:
        not_outliear_list.append(x)

start = 2 # -10 epoch
snr_res = np.loadtxt('/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results_camera/whisper_large-v1_cla.csv', delimiter=',')
snr_drop = snr_res[start] - snr_res[-1] # from -10 (3nd row, index 2) to 20 (last row) snr
print(snr_drop.shape)
snr_drop = [x*100 for x in snr_drop]

# sound classification result
all_res = []
mdl_size = 'whisper_large-v1'
for fold in range(1, 6):
    for lr in [0.001]:
        cur_res = np.loadtxt('/data/sls/scratch/yuangong/whisper-a/src/baseline_cla/baseline_res/esc_{:s}_fold{:d}_lr_{:.4f}.csv'.format(mdl_size, fold, lr), delimiter=',', usecols=list(range(6,56)))#.tolist()
        cur_res = cur_res[-2, :].tolist() # get the none-wa mean of representation, [50,] corresponds to 50 classes, -2 is the last layer out, 0 is the input layer out, -1 is the wa out
        all_res.append(cur_res)
all_res = np.array(all_res) # [5, 50] , 5 folds, 50 classes

sound_res = np.mean(all_res, axis=0) * 100
print(sound_res.shape)

print(start, 'corr', np.corrcoef(sound_res, snr_drop)[0, 1])

b, a = np.polyfit(np.array(sound_res)[not_outliear_list], np.array(snr_drop)[not_outliear_list], deg=1)

print(start, 'corr', np.corrcoef(np.array(sound_res)[not_outliear_list], np.array(snr_drop)[not_outliear_list])[0, 1])

# Create sequence of 100 numbers from 0 to 100
xseq = np.linspace(50, 100, num=50)

# Plot regression line
plt.plot(xseq, a + b * xseq, '--', lw=2.5, alpha=0.7)
plt.fill_between(xseq, a+22 + (b - 0.465) * xseq, 70, alpha=0.3, color='lightblue')

plt.scatter(sound_res, snr_drop)

font_size = 2
for cla_i in range(50):
    plt.annotate(label_list[cla_i][1:-1], (sound_res[cla_i], snr_drop[cla_i]), fontsize=font_size)

plt.ylim([-5, 110])
plt.xlim([50, 100])
plt.grid()
plt.gca().invert_yaxis()
plt.xlabel('ESC-50 Class-wise F1-Score')
plt.ylabel('Word Error Rate Increase from 20dB to -10dB SNR (%)')
figure = plt.gcf()
figure.set_size_inches(5, 5)
plt.savefig('./figure_test/snr_plot_cla_{:d}_cr_outlier.pdf'.format(start), dpi=300, bbox_inches='tight')
plt.close()