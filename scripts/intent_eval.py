"""Quality of the predicted future step-intentions on the target test sets (moving agents only).

CRF models are decoded with Viterbi, CE models with per-step argmax. Reports per-step accuracy,
macro-F1, exact-sequence match, and the label-switch rate (GT vs predicted) as a temporal-consistency
measure.

    python scripts/intent_eval.py full_traj_lam03 fullce_traj_l03 --seed 0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from stcrf.data import PAIRS, Split, load_split  # noqa: E402
from stcrf.model import STCRF, init_spaces, prepare  # noqa: E402
from train import DEFAULTS, PAIR_CFG  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')


def macro_f1(y, p, c=3):
    f = []
    for k in range(c):
        tp, fp, fn = ((p == k) & (y == k)).sum(), ((p == k) & (y != k)).sum(), ((p != k) & (y == k)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


@torch.no_grad()
def decode(exp, pair, seed, device):
    ck = torch.load(os.path.join(ROOT, 'checkpoints', exp, f'{pair}_s{seed}.pt'), weights_only=False, map_location=device)
    cfg, sd = {**DEFAULTS, **ck['cfg']}, PAIR_CFG[pair]['static_dist']
    tr, te = Split(load_split(pair, 'train'), device), Split(load_split(pair, 'test'), device)
    model = STCRF(cfg, init_spaces(tr.obs, tr.pred, sd, cfg['k'], cfg['avg2'])).to(device).eval()
    model.load_state_dict(ck['state'])
    P = prepare(te, sd, cfg['avg2'])
    ys, ps = [], []
    t_f = cfg['pred_len']
    for rows, valid, idx in te.batches(4096):
        out = model(P, rows, valid)
        e_lat, e_lon, _, _, mv = out['e']
        e = torch.stack([e_lat, e_lon], 1)
        pred = model.crf.viterbi(e) if 'crf' in cfg['intent'] else e.argmax(-1)
        gt = torch.stack([P['lat_f'][idx], P['lon_f'][idx]], 1)
        ys.append(gt[mv].cpu().numpy())
        ps.append(pred[mv][..., -t_f:].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exps', nargs='+')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    for exp in a.exps:
        res = []
        for pair in PAIRS:
            y, p = decode(exp, pair, a.seed, 'cuda')
            r = []
            for s in range(2):  # 0 lateral, 1 longitudinal
                r += [(y[:, s] == p[:, s]).mean(), macro_f1(y[:, s], p[:, s]),
                      (y[:, s] == p[:, s]).all(1).mean(),
                      (np.diff(y[:, s], axis=1) != 0).mean(), (np.diff(p[:, s], axis=1) != 0).mean()]
            res.append(r)
        m = np.mean(res, 0)
        print(f'{exp}: lateral acc {m[0]:.3f} F1 {m[1]:.3f} seq {m[2]:.3f} switch gt/pred {m[3]:.3f}/{m[4]:.3f} | '
              f'longitudinal acc {m[5]:.3f} F1 {m[6]:.3f} seq {m[7]:.3f} switch gt/pred {m[8]:.3f}/{m[9]:.3f}')
