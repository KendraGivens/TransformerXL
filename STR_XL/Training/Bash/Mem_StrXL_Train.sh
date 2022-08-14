#!/bin/sh

export WANDB_DISABLED=FALSE
export WANDB_ARTIFACTS_PATH=/data/klg6z/artifacts

python3 M_StrXL_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-8dim:latest --wandb-project "Mem_StrXL" --save_to "../../Weights/Mem_StrXL_Run1" --gpus 0

