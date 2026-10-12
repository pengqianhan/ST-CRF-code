"""Data loading for the ETH/UCY domain-adaptation pairs.

The sequence slicing is a copy of SingularTrajectory's `TrajectoryDataset`
(https://github.com/InhwanBae/SingularTrajectory, `utils/dataloader.py`; CC BY-NC 4.0), so the agents and
windows are identical to those behind SingularTrajectory's Table 1. Only the frame lookup is faster.
Processed splits are cached as .pt files and moved to the GPU once.

The datasets directory (default: ./datasets, or the environment variable STCRF_DATA) has the layout of
SingularTrajectory's `datasets/`; see README.md.
"""
import math
import os

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.abspath(os.environ.get('STCRF_DATA', os.path.join(REPO_ROOT, 'datasets')))
CACHE_ROOT = os.path.abspath(os.environ.get('STCRF_CACHE', os.path.join(REPO_ROOT, 'cache')))
PAIRS = [a + '2' + b for a in 'ABCDE' for b in 'ABCDE' if a != b]
SCENE_IMG = {'biwi_eth': 'seq_eth', 'biwi_hotel': 'seq_hotel',
             'students001': 'students003', 'students003': 'students003', 'uni_examples': 'students003',
             'crowds_zara01': 'crowds_zara01', 'crowds_zara02': 'crowds_zara02', 'crowds_zara03': 'crowds_zara02'}


def _read_file(path):
    with open(path) as f:
        return np.asarray([[float(v) for v in line.strip().split('\t')] for line in f])


def build_split(split_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):
    seq_len = obs_len + pred_len
    obs_list, pred_list, num_peds, scene_list = [], [], [], []
    for fname in sorted(os.listdir(split_dir)):
        phase = os.path.basename(os.path.normpath(split_dir))
        scene = os.path.splitext(fname)[0].replace('_' + phase, '')
        data = _read_file(os.path.join(split_dir, fname))
        frames = np.unique(data[:, 0]).tolist()
        frame_pos = {f: i for i, f in enumerate(frames)}
        frame_data = [data[data[:, 0] == f] for f in frames]
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
        for idx in range(0, num_sequences * skip + 1, skip):
            curr = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
            seqs = []
            for ped_id in np.unique(curr[:, 1]):
                ped = np.around(curr[curr[:, 1] == ped_id], decimals=4)
                pad_front = frame_pos[ped[0, 0]] - idx
                pad_end = frame_pos[ped[-1, 0]] - idx + 1
                if pad_end - pad_front != seq_len:
                    continue
                seqs.append(ped[:, 2:4])
            if len(seqs) > min_ped:
                seqs = np.stack(seqs)
                obs_list.append(seqs[:, :obs_len])
                pred_list.append(seqs[:, obs_len:])
                num_peds.append(len(seqs))
                scene_list.extend([scene] * len(seqs))
    scenes = sorted(set(scene_list))
    cum = np.cumsum([0] + num_peds)
    return {'obs': torch.from_numpy(np.concatenate(obs_list)).float(),
            'pred': torch.from_numpy(np.concatenate(pred_list)).float(),
            'seq_start_end': torch.from_numpy(np.stack([cum[:-1], cum[1:]], 1)).long(),
            'scene': torch.tensor([scenes.index(s) for s in scene_list]).long(),
            'scene_names': scenes}


def load_split(pair, phase):
    """Cached split. Val/test of the same target are shared across sources."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    path = os.path.join(CACHE_ROOT, f'{pair}_{phase}.pt')
    if not os.path.exists(path):
        d = build_split(os.path.join(DATA_ROOT, pair, phase))
        torch.save(d, path + '.tmp')
        os.replace(path + '.tmp', path)
    return torch.load(path, weights_only=False)


class Split:
    """A split on the GPU, with sequence-padded index tables for batched social attention."""

    def __init__(self, d, device):
        self.obs = d['obs'].to(device)
        self.pred = d['pred'].to(device)
        self.scene = d['scene'].to(device)
        self.scene_names = d['scene_names']
        sse = d['seq_start_end']
        self.sse = sse.numpy()
        self.num_peds = (sse[:, 1] - sse[:, 0]).numpy()
        self.n = self.obs.size(0)
        lmax = int(self.num_peds.max())
        idx = torch.full((len(sse), lmax), -1, dtype=torch.long)
        for i, (s, e) in enumerate(self.sse):
            idx[i, :e - s] = torch.arange(s, e)
        self.pad_idx = idx.to(device)
        seq_of = torch.zeros(self.n, dtype=torch.long)
        for i, (s, e) in enumerate(self.sse):
            seq_of[s:e] = i
        self.seq_of = seq_of.to(device)
        self.device = device

    def batches(self, batch_size, shuffle=False, drop_last=False, generator=None):
        """Same grouping rule as SingularTrajectory's TrajBatchSampler."""
        order = np.random.default_rng(generator).permutation(len(self.sse)) if shuffle else np.arange(len(self.sse))
        batch, total = [], 0
        for i in order:
            batch.append(i)
            total += self.num_peds[i]
            if total >= batch_size:
                yield self._make(batch)
                batch, total = [], 0
        if batch and not drop_last:
            yield self._make(batch)

    def _make(self, seq_ids):
        rows = self.pad_idx[torch.as_tensor(seq_ids, device=self.device)]
        width = int(self.num_peds[seq_ids].max())
        rows = rows[:, :width]
        valid = rows >= 0
        return rows, valid, rows[valid]
