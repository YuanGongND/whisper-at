# -*- coding: utf-8 -*-
# @Time    : 1/20/23 2:09 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : transcribe_aus.py

import sys
argument = sys.argv[1]
if argument=='4':
    argument='0,1,2,3'
import os
if argument != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"]=argument

import os
import torch
import soundfile
from transformers import Wav2Vec2Processor, HubertForCTC

def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.flac', '.wav')):
                matches.append(os.path.join(root, filename))
    return matches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-xlarge-ls960-ft").to(device)

tar_path = '/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_text_hubert_xlarge/'
if os.path.exists(tar_path) == False:
    os.mkdir(tar_path)

audio_list = fileList('/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_ready/')

audio_list.sort()
start_file = int(argument) * 4500
end_file = int(argument) * 4500 + 4500
audio_list = audio_list[start_file: end_file]

print('number of files to transcribe: ', len(audio_list))

for i in range(len(audio_list)):
    audio_path = audio_list[i]
    if os.path.exists(tar_path + audio_path.split('/')[-1].split('.')[-2] + '.txt') == False:
        if audio_path[-3:] == 'wav':
            audio_path = audio_list[i]
            source, curr_sample_rate = soundfile.read(audio_path, dtype="float32")
            input_features = processor(source,
                                       sampling_rate=curr_sample_rate,
                                       return_tensors="pt").input_values
            logits = model(input_features.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = processor.decode(predicted_ids[0])
            with open(tar_path + audio_path.split('/')[-1].split('.')[-2] + '.txt', "w") as text_file:
                text_file.write(text)
            del logits, input_features
    if i % 100 == 0:
        print("{:d} / {:d} processd from processor {:s}".format(i, len(audio_list), argument))
del model