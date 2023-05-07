#!/bin/sh
 
module load singularity

singularity exec --nv --bind /home/klg6z/work:/home/jovyan /home/jphillips/images/csci4850-2023-Spring.sif python3 StrXL_Compressed_Training.py \
    --dataset-artifact sirdavidludwig/nachusa-dna/nachusa-dna:latest \
    --encoder-artifact sirdavidludwig/dnabert-pretrain/dnabert-pretrain-64dim:latest \
    --wandb-project "StrXL_Compressed" \
    --save-to "StrXL_{seed}_{mem_len}_{num_compressed_seeds}" \
    --mem_len 250 \
    --num_compressed_seeds 50 \
    --num_induce 0 \
    --gpus 0 \
    --seed 3 \
    --epochs 1 \
    #--resume fbfb28as \