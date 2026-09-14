# ST-CRF: Enhancing Pedestrian Trajectory Prediction through Step-Intention Learning with Conditional Random Fields

This repository contains the implementation of the ST-CRF (Spatio-Temporal Conditional Random Field) model for trajectory prediction.

## Getting Started

### Prerequisites

The code has been developed and tested with:

- Python 3.11
- PyTorch 2.3.0 (CUDA 12.1)
- NumPy 1.26.4

A CUDA-capable GPU is required for training; evaluation also runs on GPU by default.

`requirements.txt` pins these versions.

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ST-CRF
   ```

2. Install PyTorch (the CUDA 12.1 build is not on PyPI, so it needs its own index):
   ```
   pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install the remaining dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Dataset Cache

The preprocessed dataset caches in `./pkls/` are stored as one zip archive per file
(~385 MB in total) so that every archive stays under GitHub's 100 MB per-file limit.
**Extract them before the first run:**

```
python pkls/extract_pkls.py
```

The script only uses the Python standard library. It skips archives that have already
been extracted (`--force` overwrites them anyway), and `--list` shows what each archive
contains without writing anything.

If you prefer to do it by hand, note that the archives are named with a leading `._`,
which makes them hidden files, so a plain `unzip *.zip` will not match anything:

```
cd pkls
find . -maxdepth 1 -name "*.zip" -exec unzip -o {} \;
cd ..
```

The extracted caches take about 1.2 GB, so make sure enough disk space is available.

Skipping this step is safe: `TrajectoryDataset` in `utils.py` rebuilds any missing cache
from `./datasets/` on first use. Rebuilding takes a few seconds for most splits and about
three minutes for `univ`, and yields byte-identical files.

### Model Checkpoints

Two sets of trained checkpoints are included:

| Directory | Contents |
| --- | --- |
| `./checkpoint_deter/` | `stcrf_eth`, `stcrf_hotel`, `stcrf_univ`, `stcrf_zara1`, `stcrf_zara2` |
| `./checkpoint_multi/` | the same five plus `stcrf_sdd`, and a `results.md` summary |

Each `stcrf_<dataset>/` directory holds `args.pkl`, `metrics.pkl`, `constant_metrics.pkl`
and `val_best.pth`.

Note that `./checkpoint/` is **not** part of the repository - it is the directory that
`train.py` creates for its own output (see [Training the Model](#training-the-model)).

### Evaluating the Model

For standard evaluation, use:

```
python test_ade_fde.py
```

This evaluates the five models in `./checkpoint_deter/` on their own test splits with
`KSTEPS = 1` (a single deterministic sample per trajectory) and prints the per-dataset
and mean ADE/FDE. To evaluate different checkpoints, edit the `paths` list near the top
of the `__main__` block.

### Zero-shot Prediction Evaluation

For zero-shot prediction evaluation, use:

```
python test_ade_fde_0shot.py
```

This script needs no editing. It loads the checkpoints from `./checkpoint_deter/` and
automatically evaluates **all 25 cross-dataset combinations** - each of the five trained
models (`eth`, `hotel`, `univ`, `zara1`, `zara2`) against every dataset it was not
trained on, including `sdd`:

- training sources: `eth`, `hotel`, `univ`, `zara1`, `zara2`
- test targets: `eth`, `hotel`, `univ`, `zara1`, `zara2`, `sdd`
- self-pairs (train == test) are skipped, leaving 5 x 6 - 5 = 25 combinations

`sdd` is only ever used as a test target, since `./checkpoint_deter/` contains no
`stcrf_sdd` model.

When the run finishes, the results are written to a timestamped markdown report in the
repository root:

```
zero_shot_results_<YYYYMMDD>_<HHMMSS>.md
```

containing a per-combination ADE/FDE table, a per-source breakdown, and the best ADE and
FDE pairs.

## Training the Model

If you want to train the ST-CRF model, run:

```
sh train.sh
```

This launches the five datasets in parallel on `CUDA_VISIBLE_DEVICES=1` (300 epochs each,
`--lr 0.01`, `--w_crfloss 0.01`). Each run writes its checkpoints to
`./checkpoint/<tag>/`, for example `./checkpoint/stcrf_eth/`.

## License

This project is licensed under the [MIT License](LICENSE).
