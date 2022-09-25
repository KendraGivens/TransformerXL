#!/bin/sh

python3 StrXL_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Gradient_StrXL" --save_to "Gradient_StrXL_Run1" --gpus 0

