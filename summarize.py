"""Summarize results/<exp>/*.json (written by evaluate.py or run_all.py): mean over seeds per pair, the
average over the pairs, and the standard deviation of that average over seeds.

    python summarize.py full_traj_lam03 full_mapf_lam03 --pairs       # cross-domain (Table 3 and 4)
    python summarize.py full_traj_lam03 none_traj --sdd --pairs       # SDD (Table 5)
    python summarize.py full_traj_lam03 full_mapf_lam03 --loo --pairs # leave-one-out (Tables 6 and 7)

With --pairs, the per-pair values are printed in the column order of the paper's tables
(E2H E2U E2Z1 E2Z2 | H2E ... | Z22Z1, i.e. A2B ... E2D with A-E = ETH, Hotel, Univ, Zara1, Zara2).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS = [a + '2' + b for a in 'ABCDE' for b in 'ABCDE' if a != b]
LOO = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
SDD = ['A2S', 'B2S', 'C2S', 'D2S', 'E2S']
SCENE = {'A': 'E', 'B': 'H', 'C': 'U', 'D': 'Z1', 'E': 'Z2', 'S': 'S'}


def label(pair):
    return pair if pair in LOO else SCENE[pair[0]] + '2' + SCENE[pair[-1]]


def load(exp):
    r = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, 'results', exp, '*.json')):
        d = json.load(open(f))
        r[d['pair']].append(d)
    return r


def stats(r, pairs):
    if any(p not in r for p in pairs):
        return None
    per = {p: (np.mean([d['ade'] for d in r[p]]), np.mean([d['fde'] for d in r[p]])) for p in pairs}
    avg = (np.mean([per[p][0] for p in pairs]), np.mean([per[p][1] for p in pairs]))
    seed_avg = []
    for s in sorted({d['seed'] for p in pairs for d in r[p]}):
        vals = [[d for d in r[p] if d['seed'] == s] for p in pairs]
        if all(vals):
            seed_avg.append((np.mean([v[0]['ade'] for v in vals]), np.mean([v[0]['fde'] for v in vals])))
    return per, avg, seed_avg


if __name__ == '__main__':
    exps = [a for a in sys.argv[1:] if not a.startswith('--')]
    show_pairs = '--pairs' in sys.argv
    pairs = LOO if '--loo' in sys.argv else (SDD if '--sdd' in sys.argv else PAIRS)
    print(f'{"experiment":20s} {"seeds":>5s}  {"average ADE/FDE":>15s}  {"std over seeds":>14s}')
    for e in exps:
        s = stats(load(e), pairs)
        if s is None:
            print(f'{e:20s} incomplete: results are missing for some pairs')
            continue
        per, avg, sa = s
        sd = np.std(np.array(sa), 0) if len(sa) > 1 else (0., 0.)
        print(f'{e:20s} {len(sa):5d}  {avg[0]:7.3f}/{avg[1]:.3f}  {sd[0]:8.3f}/{sd[1]:.3f}')
        if show_pairs:
            print('   ' + ' '.join(f'{label(p)}:{per[p][0]:.2f}/{per[p][1]:.2f}' for p in pairs))
