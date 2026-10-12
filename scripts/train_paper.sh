#!/bin/sh
# Retrain every model reported in the paper (needs GPUs; finished jobs are skipped, so the script can be
# re-run after an interruption). Adjust --gpus and --per_gpu to your machine.
cd "$(dirname "$0")/.."
GPUS="--gpus 0 --per_gpu 4"
# Tables 3 and 4: ST-CRF and the variant without step-intention supervision (5 seeds, 20 pairs)
python run_all.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf --seeds 0 1 2 3 4 $GPUS
# Table 5: SDD (5 seeds, trajectory only)
python run_all.py --exps full_traj_lam03 none_traj --seeds 0 1 2 3 4 --pairs A2S B2S C2S D2S E2S $GPUS
# Table 4: ablation variants (3 seeds)
python run_all.py --exps fullce_traj_l03 fullce_mapf_l03 obscrf_traj_l03 obscrf_mapf_l03 crf2_traj_lam03 crf2_mapf_l03 --seeds 0 1 2 $GPUS
# Table 6 and RQ5: leave-one-out (3 seeds)
python run_all.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf fullce_traj_l03 fullce_mapf_l03 obscrf_traj_l03 obscrf_mapf_l03 crf2_traj_lam03 crf2_mapf_l03 --seeds 0 1 2 --pairs eth hotel univ zara1 zara2 $GPUS
# Table 7: multi-modal leave-one-out (S = 20, 3 seeds)
python run_all.py --exps stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --seeds 0 1 2 --pairs eth hotel univ zara1 zara2 $GPUS
