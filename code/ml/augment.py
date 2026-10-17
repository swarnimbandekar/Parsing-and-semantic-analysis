"""Class-balancing strategies compared against the conditional GAN."""

import numpy as np

from .gan import CGAN


def _targets(y, n_target):
    return {c: n_target - int((y == c).sum()) for c in np.unique(y) if (y == c).sum() < n_target}


def random_oversample(X, y, n_target, rng):
    Xs, ys = [X], [y]
    for c, need in _targets(y, n_target).items():
        idx = np.flatnonzero(y == c)
        pick = rng.choice(idx, need, replace=True)
        Xs.append(X[pick])
        ys.append(np.full(need, c))
    return np.concatenate(Xs), np.concatenate(ys)


def jitter_oversample(X, y, n_target, rng, scale=0.1):
    sd = X.std(0) + 1e-6
    Xs, ys = [X], [y]
    for c, need in _targets(y, n_target).items():
        idx = np.flatnonzero(y == c)
        base = X[rng.choice(idx, need, replace=True)]
        noise = rng.normal(0, scale, base.shape) * sd
        new = np.clip(base + noise, X.min(0), X.max(0))
        Xs.append(new)
        ys.append(np.full(need, c))
    return np.concatenate(Xs), np.concatenate(ys)


def smote(X, y, n_target, rng, k=5):
    """SMOTE (Chawla et al., 2002): interpolate between a sample and a random
    one of its k nearest same-class neighbours."""
    Xs, ys = [X], [y]
    for c, need in _targets(y, n_target).items():
        idx = np.flatnonzero(y == c)
        P = X[idx]
        if len(P) == 1:
            new = np.repeat(P, need, axis=0)
        else:
            d = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d, np.inf)
            kk = min(k, len(P) - 1)
            nn = np.argsort(d, axis=1)[:, :kk]
            a = rng.integers(0, len(P), need)
            b = nn[a, rng.integers(0, kk, need)]
            lam = rng.random((need, 1))
            new = P[a] + lam * (P[b] - P[a])
        Xs.append(new)
        ys.append(np.full(need, c))
    return np.concatenate(Xs), np.concatenate(ys)


def gan_oversample(X, y, n_target, rng, loss="bce", iters=2000, seed=0, return_gan=False):
    gan = CGAN(X.shape[1], n_classes=10, loss=loss, iters=iters, seed=seed).fit(X, y)
    Xs, ys = [X], [y]
    for c, need in _targets(y, n_target).items():
        Xs.append(gan.sample(np.full(need, c)))
        ys.append(np.full(need, c))
    out = np.concatenate(Xs), np.concatenate(ys)
    return (out + (gan,)) if return_gan else out
