"""Vectorised version of SingularTrajectory's map-based anchor adjustment
(https://github.com/InhwanBae/SingularTrajectory, `anchor.py: adaptive_anchor_calculation`; CC BY-NC 4.0).

If the anchor endpoint falls outside the traversable area of the scene map, it is moved to the nearest
traversable pixel and the whole anchor is rescaled per image axis about the last observed position.
The scale factors are computed without gradient; the transform itself stays differentiable."""
import os

import numpy as np
import torch

from .data import DATA_ROOT, SCENE_IMG


def _apply_h(pts, h):
    p = torch.cat([pts, torch.ones_like(pts[..., :1])], -1) @ h.T
    return p[..., :2] / p[..., 2:]


class MapAdjuster:
    def __init__(self, scene_names, device):
        self.h, self.h_inv, self.vf, self.size = [], [], [], []
        for s in scene_names:
            h = torch.tensor(np.loadtxt(os.path.join(DATA_ROOT, 'homography', s + '_H.txt')), dtype=torch.float64)
            vf = np.load(os.path.join(DATA_ROOT, 'vectorfield', SCENE_IMG[s] + '_vector_field.npy'))
            self.h.append(h.float().to(device))
            self.h_inv.append(torch.linalg.inv(h).float().to(device))
            self.vf.append(torch.from_numpy(vf.astype(np.int32)).to(device))
            self.size.append(torch.tensor(vf.shape[1::-1], device=device) // 2)  # (x, y)

    def __call__(self, traj_w, start_w, scene):
        """traj_w (N,T,2) world, start_w (N,2) world, scene (N,) index into scene_names."""
        out = traj_w
        for s in scene.unique().tolist():
            sel = scene == s
            h, h_inv, vf, size = self.h[s], self.h_inv[s], self.vf[s], self.size[s]
            img = _apply_h(traj_w[sel], h_inv)
            with torch.no_grad():
                st = _apply_h(start_w[sel], h_inv)
                end = torch.round(img[:, -1]).long()
                end = torch.maximum(torch.minimum(end, size + size // 2 - 1), -size // 2)
                vec = vf[end[:, 1] + size[1] // 2, end[:, 0] + size[0] // 2].flip(-1).long() - size // 2
                moved = (vec != end).any(-1, keepdim=True)
                denom = end.float() - st
                scale = torch.where(denom.abs() > 1e-6, (vec.float() - st) / denom, torch.ones_like(denom))
                scale = torch.where(moved, scale, torch.ones_like(scale))
            img = (img - st[:, None]) * scale[:, None] + st[:, None]
            out = out.index_put((sel.nonzero().squeeze(1),), _apply_h(img, h))
        return out
