#!/bin/sh
module load singularity

singularity exec --nv --bind /home/klg6z/work:/home/jovyan /home/jphillips/images/csci4850-2023-Spring.sif python3 StrXL_Induced_Training.py --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest --wandb-project "StrXL_Induced" --save_to "StrXL_Induced_64_2" --gpus 0 --seed 2 --resume si6sjv3x
