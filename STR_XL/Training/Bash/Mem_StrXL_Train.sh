#!/bin/sh

python3 M_StrXL_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Mem_StrXL" --save_to "Mem_StrXL_20_0_4k_Run1" --gpus 0 --seed 0

