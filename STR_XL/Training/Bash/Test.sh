#!/bin/sh
 
module load singularity

export WANDB_MODE=offline

singularity exec --nv --writable-tmpfs --bind /home/klg6z/work/work/TransformerXL/STR_XL/Training/Cache:/home/jovyan /home/jphillips/images/csci4850-2022-Fall.sif python3 Test.py  --wandb-project "Test" --wandb-mode "offline" --save_to "Test1" --resume m0cudbaf 

