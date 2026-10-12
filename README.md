# ST-CRF: Cross-Domain Pedestrian Trajectory Prediction via Step-Intention Learning with Conditional Random Fields

[中文版](README_zh.md)

Official code and trained models of the paper *ST-CRF: Cross-Domain Pedestrian Trajectory Prediction via
Step-Intention Learning with Conditional Random Fields* (Pengqian Han, Supratim Gupta, Jiamou Liu,
Prakash Ranjitkar, Partha Roop; submitted to IEEE Access).

ST-CRF labels every observed and future step of a pedestrian trajectory with a lateral (forward / turn left /
turn right) and a longitudinal (keep speed / decelerate / accelerate) **step-intention**, computed
automatically from the trajectory. A linear-chain CRF over the step-intention sequence supervises a shared
social Transformer encoder, and the posterior marginals of the predicted steps define an
**intention-driven anchor** that a residual head refines in the low-rank singular space of
SingularTrajectory. The step-intention labels are used only for training.

## Results

Average ADE/FDE (m) with a single prediction over the 20 single-source cross-scene transfers of ETH/UCY,
evaluated with the protocol and data files of SingularTrajectory (mean over five seeds):

| Model | Parameters | Trajectory only | With scene map |
|---|---:|---:|---:|
| SingularTrajectory | 1,428,512 | 0.54/1.15 (retrained without map) | 0.48/0.94 |
| **ST-CRF** | **316,826** | **0.52/1.12** | **0.47/0.93** |

Also reported in the paper: SDD with models trained on one ETH/UCY scene (0.64/1.30), the standard
leave-one-out protocol (0.54/1.12 trajectory only, 0.48/0.95 with scene map; SingularTrajectory 0.47/0.90),
and multi-modal leave-one-out with 20 predictions (best-of-20 0.21/0.37 and 0.21/0.34).

As in SingularTrajectory, the cross-domain checkpoints are selected on the validation split of the target
scene; no file of the target scene is used for gradient updates. The leave-one-out models use no data of the
test scene at all.

## Installation

```bash
git clone https://github.com/pengqianhan/ST-CRF-code.git && cd ST-CRF-code
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.1)
```

## Data

ST-CRF reads the datasets of SingularTrajectory and the SDD files of Social-Implicit. By default the data are
expected in `./datasets`; set the environment variable `STCRF_DATA` to use another directory.

1. **ETH/UCY, cross-scene pairs, homographies, scene images and maps** (from
   [SingularTrajectory](https://github.com/InhwanBae/SingularTrajectory)):
   ```bash
   git clone https://github.com/InhwanBae/SingularTrajectory.git /tmp/SingularTrajectory
   (cd /tmp/SingularTrajectory && ./script/download_datasets.sh)
   mv /tmp/SingularTrajectory/datasets ./datasets
   ```
2. **SDD in ETH/UCY format** (from [Social-Implicit](https://github.com/abduallahmohamed/Social-Implicit)),
   needed only for Table 5:
   ```bash
   git clone https://github.com/abduallahmohamed/Social-Implicit.git /tmp/Social-Implicit
   python scripts/prepare_sdd.py --sdd /tmp/Social-Implicit/datasets/sdd
   ```

The resulting layout:

```
datasets/
  A2B/ ... E2D/          20 cross-scene pairs (A-E = ETH, Hotel, Univ, Zara1, Zara2), each with train/ val/ test/
  A2S/ ... E2S/          SDD pairs (train = source scene, val/test = SDD)
  eth/ hotel/ univ/ zara1/ zara2/   leave-one-out splits
  homography/ image/ vectorfield/   scene maps for the scene-map variant
```

Processed splits are cached in `./cache` on first use.

## Pretrained models

Download the checkpoints from the [v1.0 release](https://github.com/pengqianhan/ST-CRF-code/releases/tag/v1.0)
and unzip them in the repository root; they extract to `checkpoints/<experiment>/<pair>_s<seed>.pt`.

| File | Contents | Size |
|---|---|---:|
| `stcrf-checkpoints-main.zip` | ST-CRF and the variant without step-intention supervision, with and without scene map (Tables 3-7) | 584 MB |
| `stcrf-checkpoints-ablation.zip` | CE and CRF-on-part-of-the-sequence ablations (Table 4, RQ5) | 520 MB |
| `stcrf-checkpoints-multimodal.zip` | Multi-modal leave-one-out models (Table 7) | 74 MB |

```bash
wget https://github.com/pengqianhan/ST-CRF-code/releases/download/v1.0/stcrf-checkpoints-main.zip
unzip stcrf-checkpoints-main.zip
sha256sum -c SHA256SUMS.txt --ignore-missing   # optional, with SHA256SUMS.txt from the release
```

## Reproducing the tables from the checkpoints

`evaluate.py` evaluates every checkpoint on the test split of its pair and writes
`results/<experiment>/<pair>_s<seed>.json`; `summarize.py` averages over seeds. No training is involved.

```bash
# Table 3 (ST-CRF) and Table 4 (ablations): 20 cross-domain pairs
python evaluate.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf
python summarize.py full_traj_lam03 full_mapf_lam03 none_traj none_mapf --pairs

# Table 5: SDD
python evaluate.py --exps full_traj_lam03 none_traj --pairs A2S B2S C2S D2S E2S
python summarize.py full_traj_lam03 none_traj --sdd --pairs

# Table 6 and Table 7: leave-one-out, single and multi-modal (best-of-20)
python evaluate.py --exps full_traj_lam03 full_mapf_lam03 stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --pairs eth hotel univ zara1 zara2
python summarize.py full_traj_lam03 full_mapf_lam03 stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --loo --pairs
```

`--pairs` prints the per-pair values in the column order of the paper's tables (E2H, E2U, ..., Z22Z1).
The ablations of Table 4 are evaluated in the same way with the experiment names listed below.

| Experiment | Paper |
|---|---|
| `full_traj_lam03`, `full_mapf_lam03` | ST-CRF (trajectory only / with scene map) |
| `none_traj`, `none_mapf` | w/o step-intention supervision |
| `fullce_traj_l03`, `fullce_mapf_l03` | CE (hist. + pred.) |
| `obscrf_traj_l03`, `obscrf_mapf_l03` | CRF (hist. only) |
| `crf2_traj_lam03`, `crf2_mapf_l03` | CRF (pred. only) |
| `stoch_full_*`, `stoch_none_*` | multi-modal leave-one-out (S = 20), with / without step-intention supervision |

The configurations are defined in `experiments.py`; per-pair settings (speed threshold, batch size, learning
rate schedule) come from the released configurations of SingularTrajectory (`configs/`).

## Training

```bash
# one pair and seed
python train.py --pair A2B --seed 0 --cfg '{"intent": "full_crf", "avg2": true, "lam": 0.3}'
# experiments over pairs and seeds, several jobs per GPU; finished jobs are skipped
python run_all.py --exps full_traj_lam03 --seeds 0 1 2 3 4 --gpus 0 --per_gpu 4
# every model of the paper
sh scripts/train_paper.sh
```

`run_all.py` writes the checkpoints to `checkpoints/` and the test results to `results/`, so `summarize.py`
works on retrained models as well.

## Analysis scripts

| Script | Paper |
|---|---|
| `scripts/crf_transitions.py` | learned transition scores and Viterbi vs. per-step maximum (RQ3) |
| `scripts/fig23_values.py` | values shown in Figs. 2 and 3 |
| `scripts/intent_eval.py` | accuracy of the predicted step-intentions (RQ3) |

## Repository structure

```
stcrf/        model (model.py), linear-chain CRF (crf.py), step-intention labels (intention.py),
              normalization and singular space (geometry.py), scene-map adjustment (mapadjust.py), data (data.py)
configs/      per-pair settings from SingularTrajectory's released configurations; SDD settings
train.py      training and evaluation of one pair
run_all.py    experiments over pairs and seeds on several GPUs
evaluate.py   evaluation of saved checkpoints
summarize.py  mean over seeds per pair and on average
```

## License and acknowledgements

The trajectory normalization, singular space, data loading and scene-map adjustment follow
[SingularTrajectory](https://github.com/InhwanBae/SingularTrajectory) (Bae et al., CVPR 2024), and parts of
`stcrf/data.py`, `stcrf/geometry.py` and `stcrf/mapadjust.py` are adapted from its code. SingularTrajectory is
released under the Creative Commons Attribution-NonCommercial 4.0 International license; this repository is
released under the same license ([LICENSE](LICENSE)). The SDD files in ETH/UCY format are those of
[Social-Implicit](https://github.com/abduallahmohamed/Social-Implicit).

## Citation

```bibtex
@article{han2026stcrf,
  title   = {{ST-CRF}: Cross-Domain Pedestrian Trajectory Prediction via Step-Intention Learning with Conditional Random Fields},
  author  = {Han, Pengqian and Gupta, Supratim and Liu, Jiamou and Ranjitkar, Prakash and Roop, Partha},
  journal = {IEEE Access},
  note    = {under review},
  year    = {2026}
}
```
