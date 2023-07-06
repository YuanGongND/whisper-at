# -*- coding: utf-8 -*-
# @Time    : 3/3/23 3:04 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_noisy_speech.py

import numpy as np
import torchaudio
import torch
import os

def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.flac', '.wav')):
                matches.append(os.path.join(root, filename))
    return matches

def add_noise_torch(speech_path, noise_path, noise_db, tar_path):
    speech, sr_s = torchaudio.load(speech_path)
    noise, sr_n = torchaudio.load(noise_path)

    assert  sr_s == sr_n
    power_speech = (speech ** 2).mean()
    power_noise = (noise ** 2).mean()

    scale = (10 ** (-noise_db / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10)))

    # if speech is longer than the noise
    if speech.shape[1] > noise.shape[1]:
        ratio = int(np.ceil(speech.shape[1] / noise.shape[1]))
        noise = torch.concat([noise for _ in range(ratio)], dim=1)

    if speech.shape[1] < noise.shape[1]:
        noise = noise[:, :speech.shape[1]]

    speech = speech + scale * noise
    torchaudio.save(tar_path, speech, sample_rate=sr_s)

all_speech = fileList('/data/sls/scratch/yuangong/whisper-a/sample_audio/test-clean')
all_speech.sort()
all_speech = all_speech[:40]
print(all_speech)

all_noise_dict = {}
all_noise = fileList('/data/sls/scratch/yuangong/sslast2/egs/esc50/data/ESC-50-master/audio_16k/')
for noise in all_noise:
    cla = int(noise.split('.')[-2].split('-')[-1])
    if cla not in all_noise_dict:
        all_noise_dict[cla] = [noise]
    else:
        all_noise_dict[cla].append(noise)
print(all_noise_dict[0])
print(len(all_noise_dict[0]))

for db in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
    for cla in range(50):
        # if os.path.exists('/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_ready/{:d}/{:d}'.format(db,cla)) == False:
        #     os.makedirs('/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_ready/{:d}/{:d}'.format(db,cla))
        # for each snr, for each class, test 40 librispeech samples
        for idx in range(40):
            tar_name = '/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_ready/' + str(db) + '_' + str(cla) + '_' + all_speech[idx].split('/')[-1].split('.')[-2] + '_mix_' + all_noise_dict[cla][idx].split('/')[-1].split('.')[-2] + '.wav'
            add_noise_torch(all_speech[idx], all_noise_dict[cla][idx], db, tar_name)