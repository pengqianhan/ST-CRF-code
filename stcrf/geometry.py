"""Trajectory normalization and the Singular space, following SingularTrajectory
(https://github.com/InhwanBae/SingularTrajectory, `normalizer.py`, `space.py`; CC BY-NC 4.0),
vectorised for the whole split at once."""
import numpy as np
import torch
from scipy.interpolate import BSpline


def norm_params(obs, scale):
    """Origin = last observation; rotation from obs[-1] - obs[-3]; scale = 2 / |obs[-1] - obs[-3]|.

    Returns ori (N,2), rot (N,2,2), sca (N,). `scale` is a bool mask (moving agents get scaled)."""
    ori = obs[:, -1]
    d = obs[:, -1] - obs[:, -3]
    a = torch.atan2(d[:, 1], d[:, 0])
    rot = torch.stack([torch.stack([a.cos(), -a.sin()], 1), torch.stack([a.sin(), a.cos()], 1)], 1)
    sca = torch.where(scale, 2. / d.norm(dim=-1), torch.ones_like(a))
    return ori, rot, sca


def normalize(traj, ori, rot, sca):
    return ((traj - ori[:, None]) @ rot) * sca[:, None, None]


def denormalize(traj, ori, rot, sca):
    return (traj / sca[:, None, None]) @ rot.transpose(1, 2) + ori[:, None]


def moving_mask(obs, static_dist):
    return (obs[:, -1] - obs[:, -3]).div(2).norm(dim=-1) > static_dist


def bspline_matrix(t_obs, t_pred, dim=2, degree=2):
    """The Irwin-Hall / B-spline resampling matrix used by SingularTrajectory (space.py)."""
    twot_win, twot_hist = dim * t_pred, dim * t_obs
    steps = np.linspace(0., 1., twot_hist)
    knot = twot_win - degree + 1
    knots = np.concatenate([np.zeros(degree), np.linspace(0, 1, knot), np.ones(degree)])
    c = np.zeros([twot_hist, twot_win])
    for i in range(twot_win):
        c[:, i] = BSpline(knots, (np.arange(twot_win) == i).astype(float), degree, extrapolate=False)(steps)
    return torch.FloatTensor(c)


def singular_space(pred_norm, k, t_obs):
    """V_pred (2T_pred, k) from truncated SVD of normalized futures; V_obs = C_hist @ V_pred."""
    m = pred_norm.reshape(pred_norm.size(0), -1).T
    u, _, _ = torch.linalg.svd(m, full_matrices=False)
    v_pred = u[:, :k]
    v_obs = bspline_matrix(t_obs, pred_norm.size(1)).to(v_pred) @ v_pred
    return v_pred, v_obs
