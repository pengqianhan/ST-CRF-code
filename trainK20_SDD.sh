# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset sdd --tag stcrf_sdd  --w_crfloss 0.01 --num_epochs 300 --KSTEPS 20 && echo "eth Launched." &

