"""Learned CRF transition scores of ST-CRF and their effect on decoding (RQ3 and Figs. 2-3).

For every zero-shot model of an experiment (20 pairs x 5 seeds), on the test split of its target scene:
the largest absolute transition score, and for the moving pedestrians the mean gap between the two largest
emission scores and the fraction of steps (and of sequences) whose Viterbi label differs from the per-step
maximum of the emission scores (lateral and longitudinal chains, all L_c = 19 steps).

    python scripts/crf_transitions.py [experiment]   # default: full_traj_lam03
Writes results_analysis/crf_transitions_<experiment>.json and prints a summary per training scene.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
from stcrf.data import Split, load_split  # noqa: E402
from stcrf.model import STCRF, init_spaces, prepare  # noqa: E402
from summarize import PAIRS  # noqa: E402
from train import DEFAULTS, PAIR_CFG  # noqa: E402


def load_model(exp, pair, seed, dev='cpu'):
    ck = torch.load(os.path.join(ROOT, 'checkpoints', exp, f'{pair}_s{seed}.pt'), weights_only=False, map_location=dev)
    cfg, sd = {**DEFAULTS, **ck['cfg']}, PAIR_CFG[pair]['static_dist']
    tr = Split(load_split(pair, 'train'), dev)
    model = STCRF(cfg, init_spaces(tr.obs, tr.pred, sd, cfg['k'], cfg['avg2'])).to(dev).eval()
    model.load_state_dict(ck['state'])
    return model, cfg, sd


@torch.no_grad()
def emissions(model, cfg, sd, pair, dev='cpu'):
    """Emission scores (n, 2, L_c, 3), labels (n, 2, L_c) and moving mask of the test split."""
    te = Split(load_split(pair, 'test'), dev)
    P = prepare(te, sd, cfg['avg2'], cfg.get('label_mode', 'step'))
    E, Y, M = [], [], []
    for rows, valid, idx in te.batches(4096):
        e_lat, e_lon, y_lat, y_lon, mv = model(P, rows, valid)['e']
        E.append(torch.stack([e_lat, e_lon], 1)); Y.append(torch.stack([y_lat, y_lon], 1)); M.append(mv)
    return torch.cat(E), torch.cat(Y), torch.cat(M), te


def main(exp='full_traj_lam03'):
    torch.set_num_threads(16)
    out = []
    for pair in PAIRS:
        for seed in range(5):
            model, cfg, sd = load_model(exp, pair, seed)
            e, _, mv, _ = emissions(model, cfg, sd, pair)
            vit, arg = model.crf.viterbi(e), e.argmax(-1)
            d = vit[mv] != arg[mv]
            top2 = e[mv].topk(2, -1).values
            out.append(dict(pair=pair, seed=seed, max_abs_trans=float(model.crf.trans.abs().max()),
                            gap=float((top2[..., 0] - top2[..., 1]).mean()), steps_differ=float(d.float().mean()),
                            seqs_differ=float(d.any(-1).any(-1).float().mean()), n_moving=int(mv.sum())))
            print(out[-1], flush=True)
    path = os.path.join(ROOT, 'results_analysis', f'crf_transitions_{exp}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, 'w'), indent=1)
    for src in 'ABCDE':
        r = [x for x in out if x['pair'][0] == src]
        print(f"source {src}: median max|Psi| {np.median([x['max_abs_trans'] for x in r]):.3f}, "
              f"steps differ {100 * np.mean([x['steps_differ'] for x in r]):.2f}%, "
              f"sequences differ {100 * np.mean([x['seqs_differ'] for x in r]):.1f}%, gap {np.mean([x['gap'] for x in r]):.2f}")
    print(f"all: steps differ {100 * np.mean([x['steps_differ'] for x in out]):.2f}%, "
          f"sequences differ {100 * np.mean([x['seqs_differ'] for x in out]):.1f}%")


if __name__ == '__main__':
    main(*sys.argv[1:])
