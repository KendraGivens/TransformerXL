#!/bin/sh

python3 Str_Induced_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Str_Induced" --save_to "Str_Induced_20_Samples_Run1" --gpus 1

