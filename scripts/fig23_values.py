"""Values shown in Figs. 2 and 3: the lateral transition scores of the ST-CRF model trained on Univ and
selected on Zara1 (C2D, seed 4; its largest absolute transition score, 0.62, is close to the median of the
20 models trained on Univ, 0.72; see results_analysis/crf_transitions_full_traj_lam03.json), and the lateral
emission scores of one moving pedestrian of the Zara1 test set at the first five predicted steps. The
pedestrian was chosen so that Viterbi decoding differs from the per-step maximum on these steps; the check
below confirms that the rounded values shown in the figures give the same decodings as the full model.

    python scripts/fig23_values.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crf_transitions import emissions, load_model  # noqa: E402

PAIR, SEED, AGENT, FIRST = 'C2D', 4, 127, 7   # FIRST = chain index of T_{h+1} (7 historical steps before)
DISPLAY = [1, 0, 2]                             # turn left, forward, turn right (model order: F, L, R)
NAME = {0: 'forward', 1: 'turn left', 2: 'turn right'}


def viterbi(e, t):
    s, back = e[0].copy(), []
    for k in range(1, len(e)):
        m = s[:, None] + t
        back.append(m.argmax(0))
        s = m.max(0) + e[k]
    p = [int(s.argmax())]
    for b in reversed(back):
        p.append(int(b[p[-1]]))
    return p[::-1]


def main():
    model, cfg, sd = load_model('full_traj_lam03', PAIR, SEED)
    e, y, mv, _ = emissions(model, cfg, sd, PAIR)
    trans = model.crf.trans[0].detach().numpy()
    print('Fig. 2, lateral transition scores (rows: from, columns: to; turn left, forward, turn right):')
    print(np.round(trans[DISPLAY][:, DISPLAY], 2))
    w = e[AGENT, 0, FIRST:FIRST + 5].numpy()
    print(f'Fig. 3, agent {AGENT} moving={bool(mv[AGENT])}, emission scores at T_h+1..T_h+5 (rows: turn left, forward, turn right):')
    print(np.round(w[:, DISPLAY].T, 1))
    full = model.crf.viterbi(e[AGENT:AGENT + 1])[0, 0, FIRST:FIRST + 5].tolist()
    shown = viterbi(np.round(w, 1), np.round(trans, 2))
    print('per-step maximum:', [NAME[k] for k in w.argmax(1)])
    print('Viterbi (whole chain):', [NAME[k] for k in full], '| from the rounded values shown:', [NAME[k] for k in shown])
    assert shown == full and (np.round(w, 1).argmax(1) == w.argmax(1)).all()


if __name__ == '__main__':
    main()
