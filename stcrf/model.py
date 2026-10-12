"""ST-CRF on a SingularTrajectory-style backbone.

Per agent: normalize (origin, heading, speed), split moving/static, encode with a social transformer
over the agents of each scene window, build an anchor, add a small residual in the Singular space.

Anchor variants (`intent`):
- 'none'    : mean anchor of the training futures (SingularTrajectory with S=1, no map).
- 'crf'     : a linear-chain CRF over the 12 future step-intentions; its posterior marginals drive a
              kinematic rollout (expected turn angle and speed ratio per step) that is the anchor.
- 'ce'      : same rollout, but independent per-step softmax trained with cross-entropy.
- 'obs_crf' : CRF over the 8 observed step-intentions as an auxiliary loss only; mean anchor
              (ablation 'CRF (hist. only)').
- 'full_crf': one CRF chain over observed steps 1..7 and the 12 future steps; the observed labels
              supervise the encoder and the future marginals drive the kinematic anchor.
- 'full_ce' : same labels as 'full_crf', independent per-step softmax with cross-entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .crf import LinearChainCRF
from .geometry import denormalize, moving_mask, norm_params, normalize, singular_space
from .intention import class_magnitudes, cumulative_labels, cumulative_tables, future_labels, observed_labels

SWAP_LR = torch.tensor([0, 2, 1])


def kmeans_anchors(c, num, fallback):
    """num k-means centroids of the coefficient vectors c (N, k), as SingularTrajectory/EigenTrajectory."""
    if c.size(0) < num:
        return fallback.expand(num, -1).clone()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num, random_state=0, init='k-means++', n_init=1).fit(c.detach().cpu().numpy())
    return torch.tensor(km.cluster_centers_, dtype=c.dtype, device=c.device)


def init_spaces(obs, pred, static_dist, k, avg2=False, num_samples=1):
    """Singular spaces and mean anchors from the training split, with flip augmentation (as ST).
    With num_samples > 1, the anchors are num_samples k-means prototypes (num_samples, k)."""
    flip = torch.tensor([1., -1.], device=obs.device)
    obs_a, pred_a = torch.cat([obs, obs * flip]), torch.cat([pred, pred * flip])
    mv = moving_mask(obs_a, static_dist)
    ori, rot, sca = norm_params(obs_a, mv)
    pred_n = normalize(pred_a, ori, rot, sca)
    v_pred_m, v_obs_m = singular_space(pred_n[mv], k, obs.size(1))
    if (~mv).sum() >= 4 * k:
        v_pred_s, v_obs_s = singular_space(pred_n[~mv], k, obs.size(1))
    else:
        v_pred_s, v_obs_s = v_pred_m, v_obs_m
    anchor_m = (pred_n[mv].flatten(1) @ v_pred_m).mean(0)
    anchor_s = (pred_n[~mv].flatten(1) @ v_pred_s).mean(0) if (~mv).any() else torch.zeros_like(anchor_m)
    if num_samples > 1:
        anchor_m = kmeans_anchors(pred_n[mv].flatten(1) @ v_pred_m, num_samples, anchor_m)
        anchor_s = kmeans_anchors(pred_n[~mv].flatten(1) @ v_pred_s, num_samples, anchor_s)
    mv1 = moving_mask(obs, static_dist)
    mu_lat, log_mu_lon = class_magnitudes(obs[mv1], pred[mv1], avg2=avg2)
    mu_phi, mu_lr = cumulative_tables(pred_n[mv])  # flip-augmented moving agents
    return dict(v_pred_m=v_pred_m, v_obs_m=v_obs_m, v_pred_s=v_pred_s, v_obs_s=v_obs_s,
                anchor_m=anchor_m, anchor_s=anchor_s, mu_lat=mu_lat, log_mu_lon=log_mu_lon,
                mu_phi=mu_phi, mu_lr=mu_lr)


def prepare(split, static_dist, avg2=False, label_mode='step'):
    """Per-agent tensors that do not depend on learned weights."""
    obs, pred = split.obs, split.pred
    mv = moving_mask(obs, static_dist)
    ori, rot, sca = norm_params(obs, mv)
    seq_mean = torch.zeros(len(split.sse), 2, device=obs.device).index_add_(0, split.seq_of, ori)
    seq_mean = seq_mean / torch.as_tensor(split.num_peds, device=obs.device, dtype=torch.float)[:, None]
    pred_n = normalize(pred, ori, rot, sca)
    if label_mode == 'cum':
        lat_f, lon_f = cumulative_labels(pred_n)[:2]
    else:
        lat_f, lon_f = future_labels(obs, pred, avg2)[:2]
    lat_o, lon_o = observed_labels(obs, pred)
    return dict(obs=obs, pred=pred, moving=mv, ori=ori, rot=rot, sca=sca,
                obs_n=normalize(obs, ori, rot, sca), pred_n=pred_n,
                ori_c=ori - seq_mean[split.seq_of], scene=split.scene,
                lat_f=lat_f, lon_f=lon_f, lat_o=lat_o, lon_o=lon_o)


class STCRF(nn.Module):
    def __init__(self, cfg, spaces):
        super().__init__()
        self.cfg = cfg
        self.intent = cfg['intent']
        k, d = cfg['k'], cfg['d_model']
        for name in ['v_pred_m', 'v_obs_m', 'v_pred_s', 'v_obs_s', 'anchor_m', 'anchor_s']:
            self.register_buffer(name, spaces[name].clone())
        self.cum = cfg.get('label_mode', 'step') == 'cum'
        if self.cum:
            self.mu_phi = nn.Parameter(spaces['mu_phi'].clone())
            self.mu_lr = nn.Parameter(spaces['mu_lr'].clone())
        else:
            self.mu_lat = nn.Parameter(spaces['mu_lat'].clone())
            self.log_mu_lon = nn.Parameter(spaces['log_mu_lon'].clone())
        self.embed = nn.Sequential(nn.Linear(2 * cfg['obs_len'] + k + 3, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg['nhead'], cfg['d_ff'], dropout=cfg['dropout'], batch_first=True)
        self.social = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        if self.intent != 'none':
            full = cfg['obs_len'] - 1 + cfg['pred_len']
            self.t_int = {'obs_crf': cfg['obs_len'], 'full_crf': full, 'full_ce': full}.get(
                self.intent, cfg['pred_len'])
            self.emis = nn.Linear(d, 2 * self.t_int * 3)
            self.crf = LinearChainCRF(3, streams=2)  # lateral and longitudinal chains
        self.S = cfg.get('num_samples', 1)
        if self.S == 1:
            self.res = nn.Sequential(nn.Linear(d + k, d), nn.ReLU(), nn.Linear(d, k))
        else:  # joint residuals for the S anchors, as the SingularTrajectory denoiser
            self.res = nn.Sequential(nn.Linear(d + self.S * k, d), nn.ReLU(), nn.Linear(d, self.S * k))

    def _space(self, mv, c, basis):
        m, s = getattr(self, basis + '_m'), getattr(self, basis + '_s')
        return torch.where(mv[:, None], c @ m, c @ s)

    def _recon(self, mv, c):
        return torch.where(mv[:, None], c @ self.v_pred_m.T, c @ self.v_pred_s.T).view(c.size(0), -1, 2)

    def forward(self, P, rows, valid, flip=None, map_adj=None):
        idx = rows[valid]
        mv = P['moving'][idx]
        obs_n, pred_n, ori_c = P['obs_n'][idx], P['pred_n'][idx], P['ori_c'][idx]
        lat_f, lat_o = P['lat_f'][idx], P['lat_o'][idx]
        if flip is not None:
            f = flip[idx]
            fv = torch.stack([torch.ones_like(f), f], -1)
            obs_n, pred_n, ori_c = obs_n * fv[:, None], pred_n * fv[:, None], ori_c * fv
            swap = SWAP_LR.to(f.device)
            lat_f = torch.where(f[:, None] < 0, swap[lat_f], lat_f)
            lat_o = torch.where(f[:, None] < 0, swap[lat_o], lat_o)
        n = idx.numel()

        c_obs = self._space(mv, obs_n.flatten(1), 'v_obs')
        h = self.embed(torch.cat([obs_n.flatten(1), c_obs, ori_c, mv.float()[:, None]], 1))
        hp = h.new_zeros(*rows.shape, h.size(1))
        hp[valid] = h
        h = self.social(hp, src_key_padding_mask=~valid)[valid]

        out = {}
        anchor_m = self.anchor_m.expand(n, -1) if self.S == 1 else None
        if self.intent != 'none':
            e = self.emis(h).view(n, 2, self.t_int, 3)
            e_lat, e_lon = e[:, 0], e[:, 1]
            if self.intent == 'crf':
                y_lat, y_lon = lat_f, P['lon_f'][idx]
                nll, marg = self.crf.nll_and_marginals(e, torch.stack([y_lat, y_lon], 1))
                nll, p_lat, p_lon = nll.sum(1) / self.t_int, marg[:, 0], marg[:, 1]
            elif self.intent == 'full_crf':
                # one chain over observed steps 1..7 and the 12 future steps; observed labels supervise the
                # encoder, the marginals of the future part drive the anchor
                y_lat = torch.cat([lat_o[:, 1:], lat_f], 1)
                y_lon = torch.cat([P['lon_o'][idx][:, 1:], P['lon_f'][idx]], 1)
                nll, marg = self.crf.nll_and_marginals(e, torch.stack([y_lat, y_lon], 1))
                t_f = self.cfg['pred_len']
                nll, p_lat, p_lon = nll.sum(1) / self.t_int, marg[:, 0, -t_f:], marg[:, 1, -t_f:]
            elif self.intent == 'obs_crf':
                y_lat, y_lon = lat_o, P['lon_o'][idx]
                nll = self.crf.nll(e, torch.stack([y_lat, y_lon], 1)).sum(1) / self.t_int
            else:
                if self.intent == 'full_ce':
                    y_lat = torch.cat([lat_o[:, 1:], lat_f], 1)
                    y_lon = torch.cat([P['lon_o'][idx][:, 1:], P['lon_f'][idx]], 1)
                else:
                    y_lat, y_lon = lat_f, P['lon_f'][idx]
                nll = (F.cross_entropy(e_lat.reshape(-1, 3), y_lat.reshape(-1), reduction='none').view(n, -1).mean(1) +
                       F.cross_entropy(e_lon.reshape(-1, 3), y_lon.reshape(-1), reduction='none').view(n, -1).mean(1))
            out['nll'] = nll[mv].mean() if mv.any() else nll.sum() * 0
            out['e'] = (e_lat, e_lon, y_lat, y_lon, mv)
            if self.intent in ('crf', 'ce', 'full_crf', 'full_ce'):
                if self.intent in ('ce', 'full_ce'):
                    t_f = self.cfg['pred_len']
                    p_lat, p_lon = e_lat[:, -t_f:].softmax(-1), e_lon[:, -t_f:].softmax(-1)
                if self.S > 1:
                    out['modulation'] = (torch.cumsum(p_lat @ self.mu_lat, 1),
                                         torch.exp(torch.cumsum(p_lon @ self.log_mu_lon, 1)))
                elif self.cum:
                    # expected bearing and expected log speed ratio per step, read from per-class tables
                    phi = (p_lat * self.mu_phi).sum(-1)
                    dist = torch.exp((p_lon * self.mu_lr).sum(-1)) * torch.arange(
                        1, phi.size(1) + 1, device=phi.device, dtype=phi.dtype)
                    traj = torch.stack([dist * phi.cos(), dist * phi.sin()], -1)
                else:
                    theta = torch.cumsum(p_lat @ self.mu_lat, 1)
                    gain = torch.exp(torch.cumsum(p_lon @ self.log_mu_lon, 1))
                    s0 = (obs_n[:, -1] - obs_n[:, -3]) / 2 if self.cfg['avg2'] else obs_n[:, -1] - obs_n[:, -2]
                    c, s = theta.cos(), theta.sin()
                    step = torch.stack([c * s0[:, :1] - s * s0[:, 1:], s * s0[:, :1] + c * s0[:, 1:]], -1)
                    traj = torch.cumsum(step * gain[..., None], 1)
                if self.S == 1:
                    anchor_m = traj.flatten(1) @ self.v_pred_m
        if self.S > 1:
            return self._forward_multi(P, idx, mv, h, pred_n, flip, fv if flip is not None else None, map_adj, out)
        c_anchor = torch.where(mv[:, None], anchor_m, self.anchor_s.expand(n, -1))

        if map_adj is not None:
            # a mirrored sequence is un-mirrored, corrected on the real scene map, and mirrored back
            ori, rot, sca = P['ori'][idx], P['rot'][idx], P['sca'][idx]
            a_n = self._recon(mv, c_anchor)
            a_n = a_n * fv[:, None] if flip is not None else a_n
            a_w = map_adj(denormalize(a_n, ori, rot, sca), P['obs'][idx][:, -1], P['scene'][idx])
            a_n = normalize(a_w, ori, rot, sca)
            a_n = a_n * fv[:, None] if flip is not None else a_n
            c_anchor = self._space(mv, a_n.flatten(1), 'v_pred')

        c_pred = c_anchor + 0.1 * self.res(torch.cat([h, c_anchor], 1))
        pred_hat = self._recon(mv, c_pred)
        out['pred_n'], out['gt_n'], out['idx'] = pred_hat, pred_n, idx
        out['ade'] = ((pred_hat - pred_n).norm(dim=-1) / P['sca'][idx][:, None]).mean()
        return out

    def _forward_multi(self, P, idx, mv, h, pred_n, flip, fv, map_adj, out):
        """S > 1: k-means prototypes, each modulated step by step with the expected turn and speed change of
        the CRF (if the model has a step-intention branch); joint residuals; best-of-S loss."""
        n, S, k = idx.numel(), self.S, self.anchor_m.size(1)
        protos = (self.anchor_m @ self.v_pred_m.T).view(S, -1, 2)  # (S, T, 2), normalized frame
        if 'modulation' in out:
            theta, gain = out.pop('modulation')  # (n, T) each
            d = torch.diff(torch.cat([protos.new_zeros(S, 1, 2), protos], 1), dim=1)  # per-step displacements
            c, sn = theta.cos()[:, None], theta.sin()[:, None]  # (n, 1, T)
            dx, dy = d[None, ..., 0], d[None, ..., 1]  # (1, S, T)
            steps = torch.stack([c * dx - sn * dy, sn * dx + c * dy], -1) * gain[:, None, :, None]
            anc_m = torch.cumsum(steps, 2).flatten(2) @ self.v_pred_m  # (n, S, k)
        else:
            anc_m = self.anchor_m.expand(n, -1, -1)
        c_anchor = torch.where(mv[:, None, None], anc_m, self.anchor_s.expand(n, -1, -1))  # (n, S, k)

        mvr = mv.repeat_interleave(S)
        if map_adj is not None:
            ori, rot, sca = (P[x][idx].repeat_interleave(S, 0) for x in ('ori', 'rot', 'sca'))
            a_n = self._recon(mvr, c_anchor.reshape(n * S, k))
            fvr = fv.repeat_interleave(S, 0) if fv is not None else None
            a_n = a_n * fvr[:, None] if fvr is not None else a_n
            a_w = map_adj(denormalize(a_n, ori, rot, sca), P['obs'][idx][:, -1].repeat_interleave(S, 0),
                          P['scene'][idx].repeat_interleave(S))
            a_n = normalize(a_w, ori, rot, sca)
            a_n = a_n * fvr[:, None] if fvr is not None else a_n
            c_anchor = self._space(mvr, a_n.flatten(1), 'v_pred').view(n, S, k)

        c_pred = c_anchor + 0.1 * self.res(torch.cat([h, c_anchor.flatten(1)], 1)).view(n, S, k)
        pred_hat = self._recon(mvr, c_pred.reshape(n * S, k)).view(n, S, -1, 2)
        err = (pred_hat - pred_n[:, None]).norm(dim=-1) / P['sca'][idx][:, None, None]  # (n, S, T)
        out['pred_n'], out['gt_n'], out['idx'] = pred_hat, pred_n, idx
        out['ade'] = err.mean(-1).min(1)[0].mean()  # winner-takes-all, as SingularTrajectory
        return out
