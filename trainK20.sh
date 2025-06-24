# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset eth --tag stcrf_eth  --w_crfloss 0.01 --num_epochs 300 --KSTEPS 20 && echo "eth Launched." &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset hotel --tag stcrf_hotel  --w_crfloss 0.01 --num_epochs 300 --KSTEPS 20 && echo "hotel Launched." &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset univ --tag stcrf_univ  --w_crfloss 0.01 --num_epochs 300 --KSTEPS 20 && echo "univ Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset zara1 --tag stcrf_zara1  --w_crfloss 0.01 --num_epochs 300 --KSTEPS 20 && echo "zara1 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset zara2 --tag stcrf_zara2  --w_crfloss 0.01 --num_epochs 300 --KSTEPS 20 && echo "zara2 Launched." &
P4=$!



wait $P0 $P1 $P2 $P3 $P4