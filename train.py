"""Train and evaluate ST-CRF on one domain-adaptation pair (SingularTrajectory protocol).

train = whole source scene, val = target val (checkpoint selection by val FDE, as SingularTrajectory),
test = target test. Per-pair static_dist and batch size come from SingularTrajectory's released configs.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from stcrf.data import Split, load_split
from stcrf.geometry import denormalize
from stcrf.mapadjust import MapAdjuster
from stcrf.model import STCRF, init_spaces, prepare

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIR_CFG = {**json.load(open(os.path.join(ROOT, 'configs', 'st_domain_pairs.json'))),
            **json.load(open(os.path.join(ROOT, 'configs', 'st_deterministic_scenes.json'))),
            **json.load(open(os.path.join(ROOT, 'configs', 'sdd_pairs.json')))}  # SDD: median static_dist of the source's 4 pairs
STOCH_CFG = json.load(open(os.path.join(ROOT, 'configs', 'st_stochastic_scenes.json')))  # S = 20, leave-one-out
DEFAULTS = dict(intent='crf', use_map=False, flip=True, map_flip=False, lam=1.0, avg2=False, label_mode='step', num_samples=1, k=4, d_model=128, nhead=4, d_ff=256,
                layers=2, dropout=0.1, obs_len=8, pred_len=12, eval_batch=4096)


@torch.no_grad()
def evaluate(model, split, P, map_adj, batch=4096):
    """ADE/FDE in metres; with S > 1 samples, best-of-S ADE and FDE taken separately per agent (as ST)."""
    model.eval()
    ade, fde, n = 0., 0., 0
    for rows, valid, idx in split.batches(batch):
        out = model(P, rows, valid, map_adj=map_adj)
        pn = out['pred_n']
        if pn.dim() == 3:
            pred_w = denormalize(pn, P['ori'][idx], P['rot'][idx], P['sca'][idx])
            err = (pred_w - P['pred'][idx]).norm(dim=-1)
            ade += err.mean(1).sum().item()
            fde += err[:, -1].sum().item()
        else:
            S = pn.size(1)
            rep = lambda x: x.repeat_interleave(S, 0)
            pred_w = denormalize(pn.flatten(0, 1), rep(P['ori'][idx]), rep(P['rot'][idx]), rep(P['sca'][idx]))
            err = (pred_w.view(pn.shape) - P['pred'][idx][:, None]).norm(dim=-1)  # (n, S, T)
            ade += err.mean(-1).min(1)[0].sum().item()
            fde += err[..., -1].min(1)[0].sum().item()
        n += idx.numel()
    return ade / n, fde / n


def run(pair, cfg, seed, device='cuda', ckpt_path=None, verbose=False):
    cfg = {**DEFAULTS, **cfg}
    pc = STOCH_CFG[pair] if cfg['num_samples'] > 1 else PAIR_CFG[pair]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    tr, va, te = (Split(load_split(pair, ph), device) for ph in ('train', 'val', 'test'))
    sd = pc['static_dist']
    model = STCRF(cfg, init_spaces(tr.obs, tr.pred, sd, cfg['k'], cfg['avg2'], cfg['num_samples'])).to(device)
    P_tr, P_va, P_te = (prepare(s, sd, cfg['avg2'], cfg['label_mode']) for s in (tr, va, te))
    maps = [MapAdjuster(s.scene_names, device) if cfg['use_map'] else None for s in (tr, va, te)]
    opt = torch.optim.AdamW(model.parameters(), lr=pc['lr'], weight_decay=pc['weight_decay'])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=pc['lr_schd_step'], gamma=pc['lr_schd_gamma'])
    flip_on = cfg['flip'] and (not cfg['use_map'] or cfg['map_flip'])
    best, best_state, best_epoch, hist = float('inf'), None, -1, []
    for epoch in range(pc['num_epochs']):
        model.train()
        tot, nb = 0., 0
        for rows, valid, idx in tr.batches(pc['batch_size'], shuffle=True, drop_last=True, generator=rng):
            flip = None
            if flip_on:
                f_seq = torch.from_numpy(rng.choice([-1., 1.], size=len(tr.sse))).float().to(device)
                flip = f_seq[tr.seq_of]
            out = model(P_tr, rows, valid, flip=flip, map_adj=maps[0])
            loss = out['ade'] + (cfg['lam'] * out['nll'] if 'nll' in out else 0.)
            if torch.isnan(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), pc['clip_grad'])
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        v_ade, v_fde = evaluate(model, va, P_va, maps[1], cfg['eval_batch'])
        hist.append((tot / max(nb, 1), v_ade, v_fde))
        if v_fde < best:
            best, best_epoch = v_fde, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % 32 == 0 or epoch == pc['num_epochs'] - 1):
            print(f'{pair} ep{epoch} loss {hist[-1][0]:.4f} val {v_ade:.3f}/{v_fde:.3f} best@{best_epoch}', flush=True)
    last = evaluate(model, te, P_te, maps[2], cfg['eval_batch'])
    model.load_state_dict(best_state)
    test = evaluate(model, te, P_te, maps[2], cfg['eval_batch'])
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({'cfg': cfg, 'pair': pair, 'seed': seed, 'state': best_state}, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    return dict(pair=pair, seed=seed, cfg=cfg, ade=test[0], fde=test[1], last_ade=last[0], last_fde=last[1],
                best_epoch=best_epoch, best_val_fde=best, n_params=n_params, seconds=time.time() - t0,
                history=hist)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='A2B')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfg', default='{}', help='JSON overrides of DEFAULTS')
    ap.add_argument('--epochs', type=int, default=None)
    a = ap.parse_args()
    if a.epochs:
        for table in (PAIR_CFG, STOCH_CFG):
            if a.pair in table:
                table[a.pair]['num_epochs'] = a.epochs
    r = run(a.pair, json.loads(a.cfg), a.seed, verbose=True)
    print({k: v for k, v in r.items() if k not in ('history', 'cfg')})
