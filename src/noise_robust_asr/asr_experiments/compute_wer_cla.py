# -*- coding: utf-8 -*-
# @Time    : 3/3/23 6:27 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cal_wer.py

import os
import editdistance
import jiwer
import numpy as np


def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.txt')):
                matches.append(os.path.join(root, filename))
    return matches

def calculate_wer(seqs_hat, seqs_true):
    """Calculate sentence-level WER score.
    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level WER score
    :rtype float
    """
    word_eds, word_ref_lens = [], []
    for i in range(len(seqs_true)):
        seq_true_text = seqs_true[i]
        seq_hat_text = seqs_hat[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)

def eval_noise_wer(trans_path, result_path):
    whisper_trans = fileList(trans_path)
    truth_path = '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/ground_truth_trans/'
    truth_trans = fileList(truth_path)
    print(len(whisper_trans), len(truth_trans))

    def preprocess_text(cur_trans):
        cur_trans = jiwer.ToUpperCase()(cur_trans)
        cur_trans = jiwer.RemovePunctuation()(cur_trans)
        return cur_trans

    all_wer_list = []
    for db in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
        wer_list = []
        for cla in range(50):
            cur_trans_list, cur_truth_list = [], []
            for trans_name in whisper_trans:
                if int(trans_name.split('/')[-1].split('_')[0]) == db and int(trans_name.split('/')[-1].split('_')[1]) == cla:
                    with open(trans_name, "r") as f:
                        cur_trans = f.read()
                    cur_trans = preprocess_text(cur_trans)
                    cur_trans_list.append(cur_trans)
                    #print('trans: ', cur_trans)

                    cur_truth_name = '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/ground_truth_trans/' + trans_name.split('/')[-1].split('_mix_')[0].split('_')[2] + '.txt'
                    with open(cur_truth_name, "r") as f:
                        cur_truth = f.read()
                    cur_truth = preprocess_text(cur_truth)
                    cur_truth_list.append(cur_truth)
                    #print('truth: ', cur_truth)
            #print(len(cur_trans_list), len(cur_truth_list))
            wer = calculate_wer(cur_trans_list, cur_truth_list)
            #print('wer is ', wer)
            wer_list.append(wer)
            #print(wer_list)
        all_wer_list.append(wer_list)
        np.savetxt(result_path, all_wer_list, delimiter=',')

# eval_noise_wer('/data/sls/scratch/yuangong/whisper-a/noisy_speech_text_hubert_large/', '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results/hubert_large_cla.csv')
# eval_noise_wer('/data/sls/scratch/yuangong/whisper-a/noisy_speech_text_hubert_xlarge/', '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results/hubert_xlarge_cla.csv')
# eval_noise_wer('/data/sls/scratch/yuangong/whisper-a/noisy_speech_text_w2v_base/', '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results/w2v_base_cla.csv')
# eval_noise_wer('/data/sls/scratch/yuangong/whisper-a/noisy_speech_text_w2v_large_robust/', '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results/w2v_large_robust_cla.csv')
eval_noise_wer('/data/sls/scratch/yuangong/whisper-a/noisy_speech_camera_text_whisper_large-v1/', '/data/sls/scratch/yuangong/whisper-a/src/noisy_exp/results_camera/whisper_large-v1_cla.csv')

