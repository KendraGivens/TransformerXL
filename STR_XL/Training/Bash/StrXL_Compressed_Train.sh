#!/bin/sh

module load singularity
singularity exec --nv /home/jphillips/images/csci4850-2022-Fall.sif python3 StrXL_Compressed_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "StrXL_Compressed"  --save_to "StrXL_Compressed_500_4" --gpus 0 --seed 4

