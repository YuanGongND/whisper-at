#!/bin/bash

# need to batch to speed up processing

for((fold=0;fold<=39;fold++));
do
  sbatch extract_as_full_whisper_all.sh ${fold}
done