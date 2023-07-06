# -*- coding: utf-8 -*-
# @Time    : 1/20/23 1:35 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cluster_esc50_feat2.py

# use new all layer feat, note these feats are already pooled over time
# for whisper, w2v, and hubert models

import json
import os
os.environ["XDG_CACHE_HOME"] = './'
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def cluster_feat(dataset_json_file, tar_path):
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        num_sample = len(data)
        for idx, entry in enumerate(data):
            wav = entry["wav"]
            # the first sample
            if idx == 0:
                cur_sample = np.load(tar_path + '/' + wav.split('/')[-1][:-3] + 'npy')
                num_layer = cur_sample.shape[0]
                feat_dim = cur_sample.shape[-1]
                print('number of layers {:d} feat dim {:d}'.format(num_layer, feat_dim))
                all_feat = np.zeros((num_layer + 1, num_sample, feat_dim))
                all_label = []

            cur_rep = np.load(tar_path + '/' + wav.split('/')[-1][:-3] + 'npy')
            for layer in range(cur_rep.shape[0]):
                all_feat[layer, idx] = np.mean(cur_rep[layer], axis=0)

            all_feat[-1, idx] = np.mean(np.mean(cur_rep, axis=0), axis=0)

            cur_label = int(wav.split('.')[-2].split('-')[-1])
            all_label.append(cur_label)

    assert all_feat[0].shape[0] == len(all_label)
    return all_feat, all_label

def get_immediate_dir(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

mdl_size_list = ['wav2vec2-large-robust-ft-swbd-300h']

for mdl_size in mdl_size_list:
    for fold in range(1, 6):
        print(mdl_size)
        if 'whisper' not in mdl_size:
            tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_all/' + mdl_size + '_all_layer/'
        else:
            tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_all/' + mdl_size + '/'
        esc_train1 = '/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/datafiles/esc_train_data_' + str(fold) + '.json'
        esc_eval1 =  '/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/datafiles/esc_eval_data_' + str(fold) + '.json'
        all_tr_feat, all_tr_label = cluster_feat(esc_train1, tar_path)
        all_te_feat, all_te_label = cluster_feat(esc_eval1, tar_path)

        num_layer = all_tr_feat.shape[0]
        print(all_tr_feat.shape, all_te_feat.shape)

        for lr in [0.001]:
            all_res = []
            for layer in range(num_layer):
                cla = MLPClassifier(hidden_layer_sizes=(), learning_rate='adaptive', learning_rate_init=lr, max_iter=5000, random_state=0)
                pipe = Pipeline([('scaler', StandardScaler()), ('svc', cla)])
                pipe.fit(all_tr_feat[layer], all_tr_label)
                pred = pipe.predict(all_te_feat[layer])
                acc = accuracy_score(all_te_label, pred)
                all_acc = classification_report(all_te_label, pred, output_dict=True)
                all_acc = [all_acc[str(x)]['f1-score'] for x in range(50)]
                res = [mdl_size, fold, all_te_feat[0].shape[1], lr, layer, acc] + all_acc
                all_res.append(res)
                np.savetxt('./baseline_res/esc_{:s}_fold{:d}_lr_{:.4f}.csv'.format(mdl_size, fold, lr), all_res, delimiter=',', fmt='%s')