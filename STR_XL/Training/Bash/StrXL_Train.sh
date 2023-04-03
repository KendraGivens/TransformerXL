#!/bin/sh
 
module load singularity

singularity exec --nv --bind /home/klg6z/work:/home/jovyan /home/jphillips/images/csci4850-2023-Spring.sif python3 StrXL_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "StrXL" --save-to "StrXL_0" --gpus 0 --seed 0 --resume 23a3x7eu