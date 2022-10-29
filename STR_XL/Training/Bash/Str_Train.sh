#!/bin/sh

python3 Str_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Str" --save_to "Str_20_0_4k_Run1" --gpus 0 --seed 0 