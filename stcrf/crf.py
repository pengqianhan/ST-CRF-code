"""Batched linear-chain CRF: negative log-likelihood, posterior marginals, Viterbi.

Several independent chains (e.g. lateral and longitudinal intentions) are processed together:
emissions are (N, S, T, C) with S streams, each stream with its own transition/start/end scores.
(N, T, C) inputs are treated as a single stream.
"""
import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags=3, streams=1):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(streams, num_tags, num_tags))  # trans[s, i, j]: i -> j
        self.start = nn.Parameter(torch.zeros(streams, num_tags))
        self.end = nn.Parameter(torch.zeros(streams, num_tags))

    @staticmethod
    def _4d(e, y=None):
        if e.dim() == 3:
            return e.unsqueeze(1), (y.unsqueeze(1) if y is not None else None), True
        return e, y, False

    def _alpha(self, e):
        a = [self.start + e[:, :, 0]]
        for t in range(1, e.size(2)):
            a.append(torch.logsumexp(a[-1].unsqueeze(-1) + self.trans, dim=-2) + e[:, :, t])
        return torch.stack(a, 2)

    def _beta(self, e):
        b = [self.end.expand(e.size(0), -1, -1)]
        for t in range(e.size(2) - 1, 0, -1):
            b.append(torch.logsumexp(self.trans + (e[:, :, t] + b[-1]).unsqueeze(-2), dim=-1))
        return torch.stack(b[::-1], 2)

    def _gold(self, e, y):
        n, s = y.shape[:2]
        sidx = torch.arange(s, device=y.device)[None, :, None]
        score = self.start.expand(n, -1, -1).gather(2, y[:, :, :1]).squeeze(2)
        score = score + self.end.expand(n, -1, -1).gather(2, y[:, :, -1:]).squeeze(2)
        score = score + e.gather(3, y.unsqueeze(3)).squeeze(3).sum(2)
        return score + self.trans[sidx, y[:, :, :-1], y[:, :, 1:]].sum(2)

    def log_partition(self, e):
        e, _, sq = self._4d(e)
        z = torch.logsumexp(self._alpha(e)[:, :, -1] + self.end, dim=-1)
        return z.squeeze(1) if sq else z

    def nll(self, e, y):
        """Per-sequence NLL, shape (N, S) (or (N,) for single-stream input)."""
        e, y, sq = self._4d(e, y)
        out = torch.logsumexp(self._alpha(e)[:, :, -1] + self.end, dim=-1) - self._gold(e, y)
        return out.squeeze(1) if sq else out

    def nll_and_marginals(self, e, y):
        """One forward and one backward pass give both the NLL (if y is given) and the marginals."""
        e, y, sq = self._4d(e, y)
        a, b = self._alpha(e), self._beta(e)
        log_z = torch.logsumexp(a[:, :, -1] + self.end, dim=-1)
        nll = log_z - self._gold(e, y) if y is not None else None
        marg = torch.exp(a + b - log_z[..., None, None])
        if sq:
            return (nll.squeeze(1) if nll is not None else None), marg.squeeze(1)
        return nll, marg

    def marginals(self, e):
        return self.nll_and_marginals(e, None)[1]

    @torch.no_grad()
    def viterbi(self, e):
        e, _, sq = self._4d(e)
        score, back = self.start + e[:, :, 0], []
        for t in range(1, e.size(2)):
            score, idx = (score.unsqueeze(-1) + self.trans).max(-2)
            score = score + e[:, :, t]
            back.append(idx)
        best = [(score + self.end).argmax(-1)]
        for idx in reversed(back):
            best.append(idx.gather(2, best[-1].unsqueeze(2)).squeeze(2))
        out = torch.stack(best[::-1], 2)
        return out.squeeze(1) if sq else out
