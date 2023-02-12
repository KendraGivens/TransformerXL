#!/bin/sh

module load singularity 
singularity exec --nv /home/jphillips/images/csci4850-2022-Fall.sif python3 Str_Induced_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Str_Induced" --save_to "Str_Induced_128_4" --gpus 0 --seed 4

