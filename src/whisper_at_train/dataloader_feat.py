# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
# load from whisper feats

import csv
import json
import os.path

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
from whisper.audio import log_mel_spectrogram, pad_or_trim, load_audio

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

def preemphasis(signal,coeff=0.97):
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('Using Following Mask: {:d} Freq, {:d} Time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('Using Mix-up with Rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('Now Process ' + self.dataset)

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('Number of Classes is {:d}'.format(self.label_num))

        self.tar_path= self.audio_conf.get('tar_path')
        print('Now load features from {:s}'.format(self.tar_path))

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], data_json[i]['labels']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        return datum

    def load_rep(self, path):
        try:
            # if npy file
            if path[-3:] == 'npy':
                return np.load(path)
            elif path[-3:] == 'npz':
                return np.load(path)['arr_0']
        except:
            print('a missing file', path)
            return np.zeros((6, 25, 512)) # should only work for whisper-base model, which has missing file problem

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        if 'feat_as' in self.tar_path or 'feat_esc_pool' in self.tar_path:
            fmt = '.npz'
        else:
            fmt = '.npy'

        tar_path = self.tar_path + '/'
        if filename2 == None:
            filename = tar_path + '.'.join(filename.split('/')[-1].split('.')[:-1]) + fmt
            feat = self.load_rep(filename)
            feat = torch.Tensor(feat)

            # 25 is the time length after pooling
            if feat.shape[1] < 25:
                len_diff = 25 - feat.shape[1]
                feat = torch.nn.functional.pad(feat, (0, 0, 0, len_diff))
            else:
                feat = feat[:, :25, :]

        else:
            filename = tar_path + '.'.join(filename.split('/')[-1].split('.')[:-1]) + fmt
            feat = self.load_rep(filename)
            feat = torch.Tensor(feat)

            filename2 = tar_path + '.'.join(filename2.split('/')[-1].split('.')[:-1]) + fmt
            feat2 = self.load_rep(filename2)
            feat2 = torch.Tensor(feat2)

            if feat.shape[1] < 25:
                len_diff = 25 - feat.shape[1]
                feat = torch.nn.functional.pad(feat, (0, 0, 0, len_diff))
            else:
                feat = feat[:, :25, :]
            if feat2.shape[1] < 25:
                len_diff = 25 - feat2.shape[1]
                feat2 = torch.nn.functional.pad(feat2, (0, 0, 0, len_diff))
            else:
                feat2 = feat2[:, :25, :]
            feat = mix_lambda * feat + (1 - mix_lambda) * feat2

        return feat

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda)
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            fbank = self._wav2fbank(datum['wav'], None, 0)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set, input feat shape in [25, 1280], i.e. t-f, need to transpose
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(1, 2)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.transpose(1, 2)
        return fbank, label_indices

    def __len__(self):
        return self.num_samples