#!/bin/sh

export WANDB_DISABLED=FALSE
export WANDB_ARTIFACTS_PATH=/data/klg6z/artifacts

python3 StrXL_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/dnasamples-complete:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "Gradient_StrXL" --save_to "../../Weights/Gradient_StrXL_Run1" --gpus 0

