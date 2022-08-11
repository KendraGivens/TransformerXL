#!/bin/sh

export WANDB_DISABLED=FALSE
export WANDB_ARTIFACTS_PATH=/data/klg6z/artifacts

python3 StrXL_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-8dim:latest --wandb-project "StrXL" --save_to "../Saved_Models/StrXL_Run1s --gpus 0
