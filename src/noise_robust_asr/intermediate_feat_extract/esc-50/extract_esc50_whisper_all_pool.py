# -*- coding: utf-8 -*-
# @Time    : 1/19/23 11:35 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_esc50.py

# extract representation for all layers for whisper model, pool by 10, not include the input mel.
# save as npz to save space

import json
import torch
import os
os.environ["XDG_CACHE_HOME"] = './'
import whisper
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
                # NOTE: this use a customized whisper model for feature extraction, original whisper model does not have transcribe_audio function
                _, audio_rep = mdl.transcribe_audio(wav)
                audio_rep = audio_rep[0]
                audio_rep = torch.permute(audio_rep, (2, 0, 1)).detach().cpu().numpy()
                audio_rep = skimage.measure.block_reduce(audio_rep, (1, 10, 1), np.mean) # downsample x10 for esc, 20 for audioset
                audio_rep = audio_rep[1:]
                np.savez_compressed(tar_path + '/' + wav.split('/')[-1][:-3] + 'npz', audio_rep)

mdl_size_list = ['large-v2', 'large-v1', 'medium.en', 'medium', 'small.en', 'small', 'base.en', 'base', 'tiny.en', 'tiny'] 
for mdl_size in mdl_size_list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path = '/data/sls/scratch/yuangong/whisper-a/src/{:s}.pt'.format(mdl_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims) # NOTE: this use a customized whisper model for feature extraction, original whisper model does not have transcribe_audio function
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/' + 'whisper_' + mdl_size + '/'
    esc_train1 = '/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/datafiles/esc_train_data_1.json'
    esc_eval1 = '/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/datafiles/esc_eval_data_1.json' # esc-50 is 5-fold cross-validation, so 1st train and eval split covers all datas
    extract_audio(esc_train1, model, tar_path)
    extract_audio(esc_eval1, model, tar_path)
    del model