"""Conditional GAN over compiler-derived feature vectors.

Two training objectives are provided:
  * 'bce'     - the original non-saturating GAN loss (Goodfellow et al., 2014)
  * 'wgan-gp' - Wasserstein loss with gradient penalty (Gulrajani et al., 2017)

Both are conditioned on the error-family label (Mirza & Osindero, 2014) by
concatenating a one-hot class code to the noise vector and to the
discriminator input."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scaler:
    """Per-feature standardisation with clipping, fitted on the GAN training set."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-3
        self.lo = X.min(0)
        self.hi = X.max(0)
        return self

    def transform(self, X):
        return np.clip((X - self.mu) / self.sd, -6, 6)

    def inverse(self, Z):
        X = Z * self.sd + self.mu
        return np.clip(X, self.lo, self.hi)


class Generator(nn.Module):
    def __init__(self, z_dim, n_classes, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + n_classes, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class Discriminator(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + n_classes, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1))


class CGAN:
    def __init__(self, dim, n_classes=10, z_dim=32, hidden=128, loss="bce",
                 lr=2e-4, seed=0, iters=2000, batch=64, balanced=False):
        self.dim, self.k, self.z_dim = dim, n_classes, z_dim
        self.loss, self.lr, self.iters, self.batch = loss, lr, iters, batch
        self.balanced = balanced
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.G = Generator(z_dim, n_classes, dim, hidden)
        self.D = Discriminator(dim, n_classes, hidden)
        self.scaler = Scaler()
        self.history = []

    def _onehot(self, y):
        return F.one_hot(torch.as_tensor(y, dtype=torch.long), self.k).float()

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        Z = torch.as_tensor(self.scaler.fit(X).transform(X), dtype=torch.float32)
        C = self._onehot(y)
        n = len(X)
        y = np.asarray(y)
        by_class = [np.flatnonzero(y == k) for k in range(self.k)]
        present = [k for k in range(self.k) if len(by_class[k])]

        def draw():
            """Mini-batch indices; class-balanced draws give rare classes equal weight."""
            if not self.balanced:
                return torch.as_tensor(self.rng.integers(0, n, self.batch))
            ks = self.rng.choice(present, self.batch)
            return torch.as_tensor([self.rng.choice(by_class[k]) for k in ks])

        if self.loss == "wgan-gp":
            oG = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.9))
            oD = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.9))
            n_critic = 5
        else:
            oG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
            oD = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
            n_critic = 1
        for it in range(self.iters):
            for _ in range(n_critic):
                idx = draw()
                real, c = Z[idx], C[idx]
                z = torch.randn(self.batch, self.z_dim)
                fake = self.G(z, c).detach()
                if self.loss == "wgan-gp":
                    eps = torch.rand(self.batch, 1)
                    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
                    g = torch.autograd.grad(self.D(mix, c).sum(), mix, create_graph=True)[0]
                    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
                    lD = self.D(fake, c).mean() - self.D(real, c).mean() + 10 * gp
                else:
                    ones = torch.full((self.batch, 1), 0.9)           # one-sided label smoothing
                    lD = (F.binary_cross_entropy_with_logits(self.D(real, c), ones)
                          + F.binary_cross_entropy_with_logits(self.D(fake, c), torch.zeros(self.batch, 1)))
                oD.zero_grad()
                lD.backward()
                oD.step()
            idx = draw()
            c = C[idx]
            fake = self.G(torch.randn(self.batch, self.z_dim), c)
            if self.loss == "wgan-gp":
                lG = -self.D(fake, c).mean()
            else:
                lG = F.binary_cross_entropy_with_logits(self.D(fake, c), torch.ones(self.batch, 1))
            oG.zero_grad()
            lG.backward()
            oG.step()
            if it % 100 == 0:
                self.history.append((it, float(lD.detach()), float(lG.detach())))
        return self

    @torch.no_grad()
    def sample(self, labels):
        self.G.eval()
        labels = np.asarray(labels)
        z = torch.randn(len(labels), self.z_dim)
        out = self.G(z, self._onehot(labels)).numpy()
        return self.scaler.inverse(out)

    @torch.no_grad()
    def critic(self, X, y):
        """Discriminator logit for (X, y); higher = more 'real'."""
        self.D.eval()
        Z = torch.as_tensor(self.scaler.transform(np.asarray(X, dtype=np.float32)), dtype=torch.float32)
        return self.D(Z, self._onehot(y)).squeeze(1).numpy()
