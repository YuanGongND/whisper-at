# -*- coding: utf-8 -*-
# @Time    : 1/19/23 11:35 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_esc50.py

# extract representation for all layers for whisper model, pool by 20, not include the input mel.
# save as npz to save space

import json
import torch
import os
os.environ["XDG_CACHE_HOME"] = './'
import numpy as np
from whisper.model import Whisper, ModelDimensions
import skimage.measure
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--split", type=int, default=0, help="which split")
args = parser.parse_args()

def extract_audio(dataset_json_file, mdl, tar_path):
    if os.path.exists(tar_path) == False:
        os.mkdir((tar_path))
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        for idx, entry in enumerate(data):
            wav = entry["wav"]

            if os.path.exists(tar_path + '/' + wav.split('/')[-1][:-4] + 'npz') == False:
                _, audio_rep = mdl.transcribe_audio(wav)
                audio_rep = audio_rep[0]
                audio_rep = torch.permute(audio_rep, (2, 0, 1)).detach().cpu().numpy()
                audio_rep = skimage.measure.block_reduce(audio_rep, (1, 20, 1), np.mean)
                audio_rep = audio_rep[1:]
                if idx == 0:
                    print(audio_rep.shape)
                np.savez_compressed(tar_path + '/' + wav.split('/')[-1][:-4] + 'npz', audio_rep)
                if idx % 50 == 0:
                    print(idx)

mdl_size_list = ['medium'] # , 'large-v1', 'medium.en'
mdl_size_list = mdl_size_list[::-1]
for mdl_size in mdl_size_list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path = '/data/sls/scratch/yuangong/whisper-a/src/{:s}.pt'.format(mdl_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/' + 'whisper_' + mdl_size + '/'
    esc_train1 = '/data/sls/scratch/yuangong/whisper-a/egs/audioset/feat_extract/split_json/{:d}.json'.format(args.split)
    extract_audio(esc_train1, model, tar_path)
    del model