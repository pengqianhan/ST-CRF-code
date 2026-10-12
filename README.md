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

## Pretrained models (v1.0 release)

The [v1.0 release](https://github.com/pengqianhan/ST-CRF-code/releases/tag/v1.0) contains the checkpoints of all
models reported in the paper, in three archives. Each archive extracts to `checkpoints/<experiment>/<pair>_s<seed>.pt`.

| File | Experiments | Needed for | Size |
|---|---|---|---:|
| `stcrf-checkpoints-main.zip` | `full_traj_lam03`, `full_mapf_lam03`, `none_traj`, `none_mapf` | Tables 3, 4, 5, 6 | 584 MB |
| `stcrf-checkpoints-ablation.zip` | `fullce_*`, `obscrf_*`, `crf2_*` | Table 4, RQ5 | 520 MB |
| `stcrf-checkpoints-multimodal.zip` | `stoch_*` | Table 7 | 74 MB |
| `SHA256SUMS.txt` | checksums of the three archives | | |

Quick start, from the repository root:

```bash
URL=https://github.com/pengqianhan/ST-CRF-code/releases/download/v1.0
for f in stcrf-checkpoints-main.zip stcrf-checkpoints-ablation.zip stcrf-checkpoints-multimodal.zip SHA256SUMS.txt; do
  wget -nc "$URL/$f"
done
sha256sum -c SHA256SUMS.txt          # every line must report OK
for f in stcrf-checkpoints-*.zip; do unzip -q -n "$f"; done
ls checkpoints                       # 14 experiment folders
```

Download only the archives you need: for Tables 3, 5 and 6 the main archive is enough. With fewer archives, run
`sha256sum -c SHA256SUMS.txt --ignore-missing`.

## Reproducing the tables from the checkpoints

Prepare the datasets first (see [Data](#data)). `evaluate.py` evaluates every checkpoint on the test split of
its pair and writes `results/<experiment>/<pair>_s<seed>.json`; `summarize.py` prints the mean over seeds,
averaged over the pairs, and the standard deviation of that average over seeds. No training is involved;
a GPU is used if available. Add `--pairs` to `summarize.py` to print every pair in the column order of the
paper's tables (E2H, E2U, ..., Z22Z1).

The expected outputs below are the test results of the training runs reported in the paper; re-evaluating the
checkpoints should reproduce them up to rounding in the last digit.

**Table 3 and the first and last rows of Table 4** (main archive; 20 cross-domain pairs, 5 seeds):

```bash
python evaluate.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf
python summarize.py full_traj_lam03 full_mapf_lam03 none_traj none_mapf
```
```
experiment           seeds  average ADE/FDE  std over seeds
full_traj_lam03          5    0.519/1.119     0.004/0.011
full_mapf_lam03          5    0.472/0.934     0.003/0.002
none_traj                5    0.528/1.132     0.002/0.006
none_mapf                5    0.480/0.966     0.002/0.003
```

**Table 4, ablation rows** (ablation archive; 3 seeds):

```bash
python evaluate.py --exps fullce_traj_l03 fullce_mapf_l03 obscrf_traj_l03 obscrf_mapf_l03 crf2_traj_lam03 crf2_mapf_l03
python summarize.py fullce_traj_l03 fullce_mapf_l03 obscrf_traj_l03 obscrf_mapf_l03 crf2_traj_lam03 crf2_mapf_l03
```
```
fullce_traj_l03          3    0.516/1.110     0.002/0.009
fullce_mapf_l03          3    0.476/0.941     0.000/0.003
obscrf_traj_l03          3    0.521/1.118     0.001/0.005
obscrf_mapf_l03          3    0.474/0.946     0.001/0.003
crf2_traj_lam03          3    0.519/1.118     0.002/0.007
crf2_mapf_l03            3    0.473/0.936     0.002/0.004
```

**Table 5, SDD** (main archive; needs the SDD pairs, see [Data](#data); 5 seeds):

```bash
python evaluate.py --exps full_traj_lam03 none_traj --pairs A2S B2S C2S D2S E2S
python summarize.py full_traj_lam03 none_traj --sdd
```
```
full_traj_lam03          5    0.641/1.300     0.003/0.010
none_traj                5    0.639/1.301     0.000/0.004
```

**Table 6, leave-one-out** (main archive; 3 seeds; the ST-CRF rows are in Table 6, the variant without
step-intention supervision is quoted in RQ5):

```bash
python evaluate.py --exps full_traj_lam03 full_mapf_lam03 none_traj none_mapf --pairs eth hotel univ zara1 zara2
python summarize.py full_traj_lam03 full_mapf_lam03 none_traj none_mapf --loo
```
```
full_traj_lam03          3    0.537/1.123     0.007/0.013
full_mapf_lam03          3    0.484/0.953     0.003/0.013
none_traj                3    0.535/1.114     0.002/0.001
none_mapf                3    0.479/0.930     0.003/0.001
```

For the leave-one-out ablations, use the experiment names of the ablation archive with `--pairs eth hotel univ zara1 zara2` and `--loo`.

**Table 7, multi-modal leave-one-out, best-of-20** (multi-modal archive; 3 seeds):

```bash
python evaluate.py --exps stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --pairs eth hotel univ zara1 zara2
python summarize.py stoch_full_traj stoch_full_mapf stoch_none_traj stoch_none_mapf --loo
```
```
stoch_full_traj          3    0.214/0.367     0.002/0.006
stoch_full_mapf          3    0.205/0.344     0.002/0.003
stoch_none_traj          3    0.216/0.369     0.003/0.007
stoch_none_mapf          3    0.206/0.335     0.001/0.004
```

Experiment names and the rows of the paper:

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
