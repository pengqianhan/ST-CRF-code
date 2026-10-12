"""Step-intention labels (Alg. 1 of the paper), computed in world coordinates (metres).

Lateral: 0 forward, 1 turn left, 2 turn right. Longitudinal: 0 keep speed, 1 decelerate, 2 accelerate.
Label j compares step s_{j+1} with step s_j.
- future labels (12): s_0 = obs[-1] - obs[-2] (or the 2-step average velocity), s_1..s_12 are the future steps.
- observed labels (8): label 0 is (forward, keep); label t compares the step after
  observed frame t with the step before it, so the last one uses the first future position.
"""
import math

import torch

FRAME_DT = 0.4


def _step_labels(pts, k=0.1, d_lon=0.1, s0=None):
    s = pts[:, 1:] - pts[:, :-1]
    if s0 is not None:
        s = torch.cat([s0[:, None], s[:, 1:]], 1)
    ang = torch.atan2(s[..., 1], s[..., 0])
    dth = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, 2 * math.pi) - math.pi
    d = s.norm(dim=-1)
    v = (d[:, 1:] + d[:, :-1]) / 2 / FRAME_DT
    d_lat = torch.clamp(k * v, 0.05, 0.2)
    lat = torch.where(dth < -d_lat, 2, torch.where(dth > d_lat, 1, 0))
    ratio = d[:, 1:] / d[:, :-1].clamp_min(1e-6)
    lon = torch.where(ratio > 1 + d_lon, 2, torch.where(ratio < 1 - d_lon, 1, 0))
    return lat, lon, dth, ratio, d[:, :-1]


def future_labels(obs, pred, avg2=False):
    """avg2: s_0 is the 2-step average velocity (obs[-1] - obs[-3]) / 2, the quantity used for normalization."""
    s0 = (obs[:, -1] - obs[:, -3]) / 2 if avg2 else None
    return _step_labels(torch.cat([obs[:, -2:], pred], 1), s0=s0)


def observed_labels(obs, pred):
    lat, lon, *_ = _step_labels(torch.cat([obs, pred[:, :1]], 1))
    zero = torch.zeros_like(lat[:, :1])
    return torch.cat([zero, lat], 1), torch.cat([zero, lon], 1)


def class_magnitudes(obs, pred, min_step=0.02, avg2=False):
    """Median turn angle and median speed ratio per class on a training split."""
    lat, lon, dth, ratio, d = future_labels(obs, pred, avg2)
    ok = d > min_step
    mu_lat = torch.stack([dth[ok & (lat == c)].median() if (ok & (lat == c)).any() else dth.new_zeros(())
                          for c in range(3)])
    mu_lat[0] = 0.
    log_r = ratio.clamp(1e-3, 10).log()
    log_mu_lon = torch.stack([log_r[ok & (lon == c)].median() if (ok & (lon == c)).any() else dth.new_zeros(())
                              for c in range(3)])
    log_mu_lon[0] = 0.
    return mu_lat, log_mu_lon


def cumulative_labels(pred_n, delta=0.1, eps=0.1):
    """Cumulative future intentions in the normalized frame (origin = last observation, heading = +x,
    unit = current speed per step), where constant velocity is the point (j, 0) at step j.
    Lateral from the bearing phi_j of the future point, longitudinal from the average speed ratio
    r_j = |p_j| / j so far."""
    j = torch.arange(1, pred_n.size(1) + 1, device=pred_n.device, dtype=pred_n.dtype)[None]
    phi = torch.atan2(pred_n[..., 1], pred_n[..., 0])
    r = pred_n.norm(dim=-1) / j
    lat = torch.where(phi > delta, 1, torch.where(phi < -delta, 2, 0))
    lon = torch.where(r < 1 - eps, 1, torch.where(r > 1 + eps, 2, 0))
    return lat, lon, phi, r


def cumulative_tables(pred_n, delta=0.1, eps=0.1):
    """Per-step, per-class median bearing and median log speed ratio, shape (T, 3) each."""
    lat, lon, phi, r = cumulative_labels(pred_n, delta, eps)
    log_r = r.clamp_min(1e-3).log()
    t = pred_n.size(1)
    mu_phi = torch.zeros(t, 3, device=pred_n.device)
    mu_lr = torch.zeros(t, 3, device=pred_n.device)
    for j in range(t):
        for c in range(3):
            if (lat[:, j] == c).any():
                mu_phi[j, c] = phi[:, j][lat[:, j] == c].median()
            if (lon[:, j] == c).any():
                mu_lr[j, c] = log_r[:, j][lon[:, j] == c].median()
    return mu_phi, mu_lr
