# -*- coding: utf-8 -*-
# @Time    : 5/30/23 3:00 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : rename_state_dict.py

# rename state dict (trained with feats) to put together with whisper-at model

import torch
import os

def get_immediate_files_with_extension(directory, extension='pth'):
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(extension) and os.path.isfile(os.path.join(directory, file)):
            file_list.append(os.path.join(directory, file))
    return file_list

def replace_name(ori_mdl_path, tar_mdl_path):
    sd = torch.load(ori_mdl_path, map_location='cpu')
    mdl_key_list = sd.keys()

    whisper_at_dict = {}
    for mdl_key in mdl_key_list:
        new_mdl_key = mdl_key.replace('module.', 'at_model.')
        #print(new_mdl_key)
        whisper_at_dict[new_mdl_key] = sd[mdl_key]

    print(len(sd.keys()), len(whisper_at_dict.keys()))
    torch.save(whisper_at_dict, tar_mdl_path)

mdl_list = get_immediate_files_with_extension('/data/sls/scratch/yuangong/whisper-at/exp/')
print(mdl_list)
tar_path = '/data/sls/scratch/yuangong/whisper-at/exp/converted_to_whisper_at/'
for mdl in mdl_list:
    print(mdl)
    print('-----------------------')
    replace_name(mdl, tar_path + mdl.split('/')[-1])
