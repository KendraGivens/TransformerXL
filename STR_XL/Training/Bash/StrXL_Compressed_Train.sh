#!/bin/sh

python3 StrXL_Compressed_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "StrXL_Compressed" --save_to "StrXL_Compressed_Run2" --gpus 0

