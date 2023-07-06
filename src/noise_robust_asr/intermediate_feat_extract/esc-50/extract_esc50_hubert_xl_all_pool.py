# -*- coding: utf-8 -*-
# @Time    : 1/19/23 11:35 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_esc50.py

# extract representation for all layers from hubert xl

import json
import torch
import os
os.environ["XDG_CACHE_HOME"] = './'
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import skimage.measure

def extract_audio(dataset_json_file, model, processor, tar_path):
    if os.path.exists(tar_path) == False:
        os.mkdir((tar_path))
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        for idx, entry in enumerate(data):
            wav = entry["wav"]
            audio, sr = sf.read(wav)
            assert sr == 16000

            input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)  # Batch size 1
            audio_rep = model(input_values, output_hidden_states=True).hidden_states
            audio_rep = torch.stack(audio_rep, dim=0).squeeze(1)
            audio_rep = audio_rep.detach().cpu().numpy()
            audio_rep = skimage.measure.block_reduce(audio_rep, (1, 10, 1), np.mean)
            audio_rep = audio_rep[1:]
            np.savez_compressed(tar_path + '/' + wav.split('/')[-1][:-3] + 'npz', audio_rep)

mdl_size_list = ['facebook/hubert-xlarge-ls960-ft']
for mdl_size in mdl_size_list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    model = HubertModel.from_pretrained(mdl_size)
    model.to(device)
    model.eval()

    tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/' + mdl_size.split('/')[-1] + '/'
    esc_train1 = '/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/datafiles/esc_train_data_1.json'
    esc_eval1 = '/data/sls/scratch/yuangong/whisper-a/egs/esc-50/feat_extract/data/datafiles/esc_eval_data_1.json'
    extract_audio(esc_train1, model, processor, tar_path)
    extract_audio(esc_eval1, model, processor, tar_path)
    del model