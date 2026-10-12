"""Run experiments over the 20 pairs x seeds on several GPUs, many jobs per GPU.

    python run_all.py --exps full_traj_lam03 none_traj --seeds 0 --gpus 0 1 --per_gpu 4

Each job writes results/<exp>/<pair>_s<seed>.json and checkpoints/<exp>/<pair>_s<seed>.pt.
Finished jobs are skipped, so the command can be re-run after an interruption.
"""
import argparse
import json
import os
import queue
import time
import traceback

import torch.multiprocessing as mp

from experiments import EXPERIMENTS

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS = [a + '2' + b for a in 'ABCDE' for b in 'ABCDE' if a != b]
COST = {'C': 0, 'E': 1, 'D': 2, 'B': 3, 'A': 4}  # longest sources first (Univ has 24k agents)


def worker(gpu, jobs, done):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from train import run
    while True:
        try:
            exp, pair, seed = jobs.get_nowait()
        except queue.Empty:
            return
        out = os.path.join(ROOT, 'results', exp, f'{pair}_s{seed}.json')
        try:
            r = run(pair, EXPERIMENTS[exp], seed, device=f'cuda:{gpu}',
                    ckpt_path=os.path.join(ROOT, 'checkpoints', exp, f'{pair}_s{seed}.pt'))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(r, open(out + '.tmp', 'w'))
            os.replace(out + '.tmp', out)
            done.put(f'{exp} {pair} s{seed}: {r["ade"]:.3f}/{r["fde"]:.3f} ({r["seconds"]:.0f}s, gpu{gpu})')
        except Exception:
            done.put(f'FAILED {exp} {pair} s{seed}\n{traceback.format_exc()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='*', default=[])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--pairs', nargs='+', default=PAIRS)
    ap.add_argument('--gpus', nargs='+', type=int, default=[0])
    ap.add_argument('--per_gpu', type=int, default=6)
    ap.add_argument('--extra', nargs='*', default=[], help='additional exp:seed combinations')
    a = ap.parse_args()
    combos = [(e, s) for e in a.exps for s in a.seeds] + [(x.split(':')[0], int(x.split(':')[1])) for x in a.extra]
    todo = [(e, p, s) for e, s in combos for p in a.pairs
            if not os.path.exists(os.path.join(ROOT, 'results', e, f'{p}_s{s}.json'))]
    todo.sort(key=lambda j: COST.get(j[1][0], 5))
    ctx = mp.get_context('spawn')
    jobs, done = ctx.Queue(), ctx.Queue()
    for j in todo:
        jobs.put(j)
    procs = [ctx.Process(target=worker, args=(g, jobs, done)) for g in a.gpus for _ in range(a.per_gpu)]
    t0 = time.time()
    for p in procs:
        p.start()
    for i in range(len(todo)):
        print(f'[{i + 1}/{len(todo)} {time.time() - t0:.0f}s] {done.get()}', flush=True)
    for p in procs:
        p.join()
