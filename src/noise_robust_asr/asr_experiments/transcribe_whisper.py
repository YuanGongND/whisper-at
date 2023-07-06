# -*- coding: utf-8 -*-
# @Time    : 1/20/23 2:09 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : transcribe_aus.py

# transcribe adress-m datasets
import sys
argument = sys.argv[1]
if argument=='4':
    argument='0,1,2,3'
import os
if argument != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"]=argument

os.environ["XDG_CACHE_HOME"] = './'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import whisper
import torchaudio

def show_twod(input):
    return "{:.2f}".format(input)

def get_immediate_files(a_dir):
    return [a_dir + '/' + name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.flac', '.wav')):
                matches.append(os.path.join(root, filename))
    return matches

audio_list = fileList('/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_ready/')

audio_list.sort()

print('number of files to transcribe: ', len(audio_list))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

whisper_mdl_list = ['base.en'] # medium.en, small.en, base.en, tiny.en, medium, large-v1

for mdl_size in whisper_mdl_list:
    model = whisper.load_model(mdl_size, device)
    for beam_size in [0]:
        tar_path = '/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_text_whisper_{:s}/'.format(mdl_size)
        if os.path.exists(tar_path) == False:
            os.mkdir(tar_path)
        for i in range(len(audio_list)):
            audio_path = audio_list[i]
            if os.path.exists(tar_path + audio_path.split('/')[-1].split('.')[-2] + '.txt') == False:
                if audio_path[-3:] == 'wav':
                    ori_waveform, sr = torchaudio.load(audio_path)
                    wav_len = ori_waveform.shape[1]
                    assert sr == 16000
                    if beam_size == 0:
                        result = model.transcribe(audio_path, language='en')
                    else:
                        result = model.transcribe(audio_path, beam_size=beam_size)

                    # remove the first space
                    text = result["text"][1:]
                    if os.path.exists(tar_path) == False:
                        os.mkdir(tar_path)
                    with open(tar_path + audio_path.split('/')[-1].split('.')[-2] + '.txt',  "w") as text_file:
                        text_file.write(text)
            if i % 100 == 0:
                print("{:d} / {:d} processd from processor {:s}".format(i, len(audio_list), argument))
    del model
