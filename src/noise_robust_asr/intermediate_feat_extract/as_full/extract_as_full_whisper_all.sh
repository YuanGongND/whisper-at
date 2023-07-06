#!/bin/bash
##SBATCH -p 1080,sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2],sls-sm-[1,2,11,13]
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -n 1

#SBATCH --mem=24000
#SBATCH --job-name="as_extract"
#SBATCH --output=../../log/%j_as_extract.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source /data/sls/scratch/yuangong/whisper-a/venv-wa/bin/activate
export TORCH_HOME=../../pretrained_models

python extract_as_full_whisper_all.py --split $1