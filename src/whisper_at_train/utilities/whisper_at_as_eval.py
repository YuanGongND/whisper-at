# -*- coding: utf-8 -*-
# @Time    : 5/28/23 2:36 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : whisper_transcribe_test.py

# evaluate whisper-at on as-eval set
# note this use 30s window, performance will be slightly lower than that using 10s window

import sys
argument = sys.argv[1]
if argument=='4':
    argument='0,1,2,3'
import os
if argument != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"]=argument

import whisper_at as whisper
import numpy as np
import csv
import json
import torch

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

mdl_size='large-v1'
at_low_compute=False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model(mdl_size, at_low_compute=at_low_compute).to(device)

index_dict = make_index_dict('/data/sls/scratch/yuangong/whisper-at/src/class_labels_indices.csv')
with open('/data/sls/scratch/yuangong/whisper-at/src/eval_data.json') as json_file:
    data = json.load(json_file)['data']
num_file = len(data)

all_pred, all_truth = torch.zeros([num_file, 527]).to(device), torch.zeros([num_file, 527]).to(device)
for i, entry in enumerate(data):
    cur_wav = entry['wav']
    labels = entry['labels'].split(',')
    for label in labels:
        all_truth[i, int(index_dict[label])] = 1.0
    result = model.transcribe(cur_wav, language='en', logprob_threshold=None, compression_ratio_threshold=None)['audio_tag']
    all_pred[i] = result[0]

    if i % 100 == 0:
        print(i)
        np.save('./at_res/all_pred_{:s}_low_{:s}.npy'.format(mdl_size, str(at_low_compute)), all_pred.cpu().numpy())
        np.save('./at_res/all_truth_{:s}_low_{:s}.npy'.format(mdl_size, str(at_low_compute)), all_truth.cpu().numpy())

np.save('./at_res/all_pred_{:s}_low_{:s}.npy'.format(mdl_size, str(at_low_compute)), all_pred.cpu().numpy())
np.save('./at_res/all_truth_{:s}_low_{:s}.npy'.format(mdl_size, str(at_low_compute)), all_truth.cpu().numpy())