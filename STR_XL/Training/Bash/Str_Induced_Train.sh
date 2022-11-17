#!/bin/sh

#singularity exec --nv /nfshome/jphillips/images/csci4850-2022-Fall.sif 
python3 Str_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Str_Induced" --save_to "Str_Induced_32_0_Run1" --gpus 0 --seed 0

