#!/bin/sh

python3 StrXL_Induced_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "StrXL_Induced" --save_to "StrXL_Induced_20_Samples_Run1" --gpus 0
