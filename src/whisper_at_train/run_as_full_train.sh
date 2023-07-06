#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/whisper-a/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

lr=5e-5
freqm=0
timem=10
mixup=0.5
batch_size=48
model=whisper-high-lw_tr_1_8 #whisper-high-lw_tr_1_8 (tl-tr, lr=5e-5) whisper-high-lw_down_tr_512_1_8 (tl-tr-512, w/ low-dim proj, lr=1e-4)
model_size=large-v2

dataset=as-full
bal=bal
epoch=30
lrscheduler_start=15
lrscheduler_decay=0.75
lrscheduler_step=5
wa=True
wa_start=16
wa_end=30
lr_adapt=False
tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
te_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/eval_data.json
label_smooth=0.1

exp_dir=./exp/test-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-bs${batch_size}-lda${lr_adapt}-mix${mixup}-${freqm}-${timem}
mkdir -p $exp_dir

python -W ignore ./run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /data/sls/scratch/yuangong/convast/egs/audioset/data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--model_size ${model_size} --label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--loss BCE --metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--num-workers 8