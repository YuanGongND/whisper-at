# -*- coding: utf-8 -*-
# @Time    : 6/1/23 3:33 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : at_post_processing.py

import os
import json
import torch
import warnings
from .tokenizer import LANGUAGES

def parse_at_label(result, language='follow_asr', top_k=5, p_threshold=-1, include_class_list=list(range(527))):
    """
    :param result: The result dict returned by the whisper-at transcribe function.
    :param language: The audio tag label name language, e.g., 'en', 'zh'. Default='follow_asr', i.e., same with ASR result.
    :param top_k: Output up to k sound classes that have logits above p_threshold. Default=5.
    :param p_threshold: The logit threshold to predict a sound class. Default=-1.
    :param p_threshold: A list of indexes that of interest. Default = list(range(527)) (all classes).
    :return: A dictionary of audio tagging results
    """
    asr_language = result['language']
    at_time_res = result['at_time_res']
    audio_tag = result['audio_tag']

    if language == 'follow_asr':
        language = asr_language

    with open(os.path.join(os.path.dirname(__file__), "assets", "label_name_dict.json")) as json_file:
        label_name_dict = json.load(json_file)

    if language not in label_name_dict.keys():
        warnings.warn("{:s} language not supported. Use English label names instead. If you wish to use label names of a specific language, please specify the language argument".format(language))
        language = 'en'

    label_name_list = label_name_dict[language]

    all_res = []
    for i in range(audio_tag.shape[0]):
        top_values, top_indices = torch.topk(audio_tag[i], k=top_k)
        cur_time_stamp = {'start': i*at_time_res, 'end': (i+1)*at_time_res}
        cur_labels_list = []
        for j in range(top_indices.shape[0]):
            if top_values[j] > p_threshold and top_indices[j] in include_class_list:
                cur_label = (label_name_list[top_indices[j]], top_values[j].item())
                cur_labels_list.append(cur_label)
        all_res.append({'time': cur_time_stamp, 'audio tags': cur_labels_list})
    return all_res

def print_label_name(language='en'):
    with open(os.path.join(os.path.dirname(__file__), "assets", "label_name_dict.json")) as json_file:
        label_name_dict = json.load(json_file)
    label_name_list = label_name_dict[language]
    for i in range(len(label_name_list)):
        print("index: {:d} : {:s}".format(i, label_name_list[i]))

def print_support_language():
    with open(os.path.join(os.path.dirname(__file__), "assets", "label_name_dict.json")) as json_file:
        label_name_dict = json.load(json_file)
    for key in label_name_dict.keys():
        print("language code: {:s} : {:s}".format(key, LANGUAGES[key]))

if __name__ == '__main__':
    print_support_language()
    print_label_name(language='zh')

