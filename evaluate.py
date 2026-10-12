"""Evaluate released (or newly trained) ST-CRF checkpoints on the test split of their pair.

    python evaluate.py --exps full_traj_lam03 full_mapf_lam03                  # 20 cross-domain pairs
    python evaluate.py --exps full_traj_lam03 none_traj --pairs A2S B2S C2S D2S E2S   # SDD
    python evaluate.py --exps full_traj_lam03 --pairs eth hotel univ zara1 zara2      # leave-one-out

For every checkpoints/<exp>/<pair>_s<seed>.pt found, writes results/<exp>/<pair>_s<seed>.json with the test
ADE/FDE (best-of-20 for the stochastic models), which summarize.py averages over seeds. The model is rebuilt
from the training split of the pair (Singular space, anchors, class medians) and then loaded from the
checkpoint, so the datasets must be prepared as described in README.md. No training is done.
"""
import argparse
import glob
import json
import os

import torch

from run_all import PAIRS
from stcrf.data import Split, load_split
from stcrf.mapadjust import MapAdjuster
from stcrf.model import STCRF, init_spaces, prepare
from train import DEFAULTS, PAIR_CFG, STOCH_CFG, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg, pair = {**DEFAULTS, **ck['cfg']}, ck['pair']
    sd = (STOCH_CFG[pair] if cfg['num_samples'] > 1 else PAIR_CFG[pair])['static_dist']
    tr = Split(load_split(pair, 'train'), device)
    te = Split(load_split(pair, 'test'), device)
    model = STCRF(cfg, init_spaces(tr.obs, tr.pred, sd, cfg['k'], cfg['avg2'], cfg['num_samples'])).to(device)
    model.load_state_dict(ck['state'])
    P_te = prepare(te, sd, cfg['avg2'], cfg['label_mode'])
    map_adj = MapAdjuster(te.scene_names, device) if cfg['use_map'] else None
    ade, fde = evaluate(model, te, P_te, map_adj, cfg['eval_batch'])
    return dict(pair=pair, seed=ck['seed'], cfg=cfg, ade=ade, fde=fde,
                n_params=sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='+', required=True)
    ap.add_argument('--pairs', nargs='+', default=PAIRS)
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='default: all seeds found')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    for exp in a.exps:
        for pair in a.pairs:
            for path in sorted(glob.glob(os.path.join(ROOT, 'checkpoints', exp, f'{pair}_s*.pt'))):
                seed = int(os.path.basename(path)[len(pair) + 2:-3])
                if a.seeds is not None and seed not in a.seeds:
                    continue
                r = evaluate_checkpoint(path, a.device)
                out = os.path.join(ROOT, 'results', exp, f'{pair}_s{r["seed"]}.json')
                os.makedirs(os.path.dirname(out), exist_ok=True)
                json.dump(r, open(out, 'w'))
                print(f'{exp} {pair} s{r["seed"]}: ADE/FDE {r["ade"]:.3f}/{r["fde"]:.3f} '
                      f'({r["n_params"]:,} parameters)', flush=True)
