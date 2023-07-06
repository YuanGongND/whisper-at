# -*- coding: utf-8 -*-
# @Time    : 5/28/23 2:36 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : whisper_transcribe_test.py

import whisper_at as whisper

model = whisper.load_model("large-v1")
result = model.transcribe("/data/sls/scratch/yuangong/dataset/adress_train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S024.wav")
#result = model.transcribe("/data/sls/scratch/yuangong/whisper-at/sample_audio/007P6bFgRCU_10.000.flac", at_time_res=2)

print(result['text'])
print(result['segments'])
text_segments = result['segments']
text_annotation = [(x['start'], x['end'], x['text']) for x in text_segments]
print(text_annotation)
at_res = whisper.parse_at_label(result, language='en', p_threshold=-1)
print(at_res)

all_seg = []
for segment in at_res:
    cur_start = segment['time']['start']
    cur_end = segment['time']['end']
    cur_tags = segment['audio tags']
    cur_tags = [x[0] for x in cur_tags]
    cur_tags = '; '.join(cur_tags)
    all_seg.append((cur_start, cur_end, cur_tags))
print(all_seg)

whisper.print_support_language()
whisper.print_label_name(language='zh')