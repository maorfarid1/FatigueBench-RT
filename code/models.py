"""Prognostic models for FatigueBench-RT.

Every predict() returns (point_rul [n], samples_rul [n, S]) for the stacked evaluation points.
"""
import numpy as np, torch, torch.nn as nn
from features import dimensionless
from scipy.stats import norm, truncnorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import os
torch.set_num_threads(int(os.environ.get("NT", "2")))
S_SAMPLES = 200
LOG2PI = np.log(2 * np.pi)


def stack(data, key=None):
    """Stack valid points of all trajectories. Returns X, aux arrays, traj index."""
    Xs, idx, cols = [], [], {}
    for i, (X, a) in enumerate(data):
        v = a["valid"]
        Xs.append(X[v]); idx.append(np.full(v.sum(), i))
        for k in ["rul", "dD", "rate_future", "D", "D_dk", "r_fast", "r_slow", "r_dk", "tt", "lr_mean", "lr_var"]:
            cols.setdefault(k, []).append(a[k][v])
        cols.setdefault("Kf", []).append(np.full(v.sum(), a["Kf"]))
        cols.setdefault("w", []).append(np.full(v.sum(), 1.0 / v.sum()))
        cols.setdefault("farid", []).append(a["farid"][v])
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    return np.concatenate(Xs), cols, np.concatenate(idx)


class Scaler:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


def mlp(i, o, h=64, p=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h), nn.SiLU(),
                         nn.Dropout(p), nn.Linear(h, o))


def gnll(y, mu, logv):
    return 0.5 * (logv + (y - mu) ** 2 / torch.exp(logv))


def train_loop(model, loss_fn, n, epochs, seed, bs=512, lr=2e-3, val_fn=None):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            opt.zero_grad()
            loss = loss_fn(b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if val_fn is not None and (ep % 5 == 4 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                v = val_fn()
            if v < best:
                best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def mixture_samples(mu, sd, rng):
    """mu, sd: [n, M] Gaussian mixture in log-RUL -> RUL samples [n, S]."""
    n, M = mu.shape
    comp = rng.integers(0, M, size=(n, S_SAMPLES))
    m = np.take_along_axis(mu, comp, 1)
    s = np.take_along_axis(sd, comp, 1)
    return np.exp(np.clip(m + s * rng.standard_normal((n, S_SAMPLES)), -15.0, 15.0))


# ----------------------------------------------------------------------------------------------
class MinerExtrapolation:
    """Real-time Palmgren-Miner extrapolation: RUL = (1 - D_hat) / r_hat, lognormal scatter from val."""
    name = "Miner-RT"

    def __init__(self, rate_key="r_slow", name=None, dirlik=False):
        self.rate_key, self.dirlik = rate_key, dirlik
        if name: self.name = name

    def _mu(self, c):
        if self.dirlik:   # cumulative damage and rate from Dirlik's spectral estimate
            D, r = c["D_dk"], c["D_dk"] / c["tt"]
        else:
            D, r = c["D"], c["D"] / c["tt"]      # accumulated damage and mean damage rate so far
        return np.log(np.maximum(1.0 - D, 0.02) / np.maximum(r, 1e-12))

    def fit(self, tr, va):
        _, c, _ = va
        self.sd = np.std(np.log(c["rul"]) - self._mu(c))
        return self

    def predict(self, te, rng):
        _, c, _ = te
        mu = self._mu(c)[:, None]
        return np.exp(mu[:, 0]), mixture_samples(mu, np.full_like(mu, self.sd), rng)


# ----------------------------------------------------------------------------------------------
class Farid2022:
    """Re-implementation of the hybrid FC-ANN + GPR model with Bayesian survival truncation (Farid 2022).

    Inputs: material (A, b, sigma_uts), running mean and running 90th-percentile amplitude. The ANN
    (5-12-1) predicts log total failure time, a GPR maps the ANN output to a Gaussian over log failure
    time, and the posterior is truncated at the current time (Heaviside likelihood). Log time is used
    instead of linear time for numerical robustness over two decades of lives.
    """
    name = "ANN+GPR (Farid 2022)"

    def fit(self, tr, va, seed=0):
        Xtr, ctr, _ = tr
        self.sc = Scaler().fit(ctr["farid"])
        X = torch.tensor(self.sc(ctr["farid"]), dtype=torch.float32)
        y = torch.tensor(np.log(ctr["Kf"]), dtype=torch.float32)
        w = torch.tensor(ctr["w"] / ctr["w"].mean(), dtype=torch.float32)
        self.ym = float(y.mean())
        net = nn.Sequential(nn.Linear(5, 12), nn.Tanh(), nn.Linear(12, 1))
        Xv = torch.tensor(self.sc(va[1]["farid"]), dtype=torch.float32)
        yv = torch.tensor(np.log(va[1]["Kf"]), dtype=torch.float32)
        loss = lambda b: (w[b] * (net(X[b]).squeeze(-1) + self.ym - y[b]) ** 2).mean()
        vfn = lambda: float(((net(Xv).squeeze(-1) + self.ym - yv) ** 2).mean())
        self.net = train_loop(net, loss, len(y), 60, seed, val_fn=vfn)
        # GPR on (eta, log tau) from the validation (CV) set, as in the original paper
        with torch.no_grad():
            eta = self.net(Xv).squeeze(-1).numpy() + self.ym
        rs = np.random.default_rng(seed)
        sel = rs.choice(len(eta), size=min(1500, len(eta)), replace=False)
        k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        self.gpr = GaussianProcessRegressor(k, normalize_y=True, random_state=seed)
        self.gpr.fit(eta[sel, None], yv.numpy()[sel])
        return self

    def predict(self, te, rng):
        _, c, _ = te
        with torch.no_grad():
            eta = self.net(torch.tensor(self.sc(c["farid"]), dtype=torch.float32)).squeeze(-1).numpy() + self.ym
        mu, sd = self.gpr.predict(eta[:, None], return_std=True)
        lt = np.log(c["tt"])
        a = (lt - mu) / sd
        samples_logT = truncnorm.rvs(a[:, None], np.inf, loc=mu[:, None], scale=sd[:, None],
                                     size=(len(mu), S_SAMPLES), random_state=rng.integers(1 << 31))
        T_map = np.exp(np.maximum(mu, lt))                    # mode of truncated posterior
        rul_s = np.maximum(np.exp(samples_logT) - c["tt"][:, None], 1e-3)
        point = np.maximum(T_map - c["tt"], 1e-3)
        point = np.where(point <= 1e-3, np.median(rul_s, 1), point)
        return point, rul_s


# ----------------------------------------------------------------------------------------------
class DeepEnsembleMLP:
    """Black-box heteroscedastic deep ensemble on the full real-time feature vector (log RUL)."""
    name = "Deep ensemble MLP"

    def __init__(self, M=5, epochs=60):
        self.M, self.epochs = M, epochs

    def fit(self, tr, va, seed=0):
        Xtr, c, _ = tr
        self.sc = Scaler().fit(Xtr)
        X = torch.tensor(self.sc(Xtr), dtype=torch.float32)
        y = torch.tensor(np.log(c["rul"]), dtype=torch.float32)
        w = torch.tensor(c["w"] / c["w"].mean(), dtype=torch.float32)
        Xv = torch.tensor(self.sc(va[0]), dtype=torch.float32)
        yv = torch.tensor(np.log(va[1]["rul"]), dtype=torch.float32)
        self.ym = float(y.mean())
        self.nets = []
        for m in range(self.M):
            net = mlp(X.shape[1], 2)
            def loss(b, net=net):
                o = net(X[b]); return (w[b] * gnll(y[b], o[:, 0] + self.ym, o[:, 1])).mean()
            def vfn(net=net):
                o = net(Xv); return float(gnll(yv, o[:, 0] + self.ym, o[:, 1]).mean())
            self.nets.append(train_loop(net, loss, len(y), self.epochs, seed * 10 + m, val_fn=vfn))
        return self

    def dist(self, X):
        Xt = torch.tensor(self.sc(X), dtype=torch.float32)
        mus, sds = [], []
        with torch.no_grad():
            for net in self.nets:
                net.eval(); o = net(Xt)
                mus.append(o[:, 0].numpy() + self.ym); sds.append(np.exp(0.5 * o[:, 1].numpy()))
        return np.stack(mus, 1), np.stack(sds, 1)

    def predict(self, te, rng):
        mu, sd = self.dist(te[0])
        return np.exp(mu.mean(1)), mixture_samples(mu, sd, rng)


# ----------------------------------------------------------------------------------------------
class PIBNN(DeepEnsembleMLP):
    """Physics-informed Bayesian neural network (deep-ensemble approximation).

    log RUL = log(Delta D) - log(r) with
      log(Delta D) = log(max(1 - D_hat, 0.02)) + f_D(z)      remaining damage to failure (Miner prior + residual)
      log(r)       = log(D_hat / t)           + f_r(z)      future damage rate (mean-rate prior + learned forecast)
    The residual networks f_D, f_r only see scale-free inputs z (damage fraction, rate ratios, kurtosis, ...),
    so the prediction is equivariant to the absolute damage-rate/life scale and extrapolates in life.
    Both factors are supervised (damage-state and load-forecast losses) and carry their own
    heteroscedastic variances; Var[log RUL] = Var_D + Var_r. RUL > 0 and monotone damage hold by construction.
    """
    name = "PI-BNN (proposed)"

    def __init__(self, M=5, epochs=60, lam=1.0, use_physics_prior=True, use_aux=True, rate_prior="mean",
                 scale_free=True, shrink=True, name=None):
        super().__init__(M, epochs)
        self.lam, self.prior, self.aux, self.rate_prior, self.sf = lam, use_physics_prior, use_aux, rate_prior, scale_free
        self.shrink = shrink and bool(rate_prior)
        if name: self.name = name

    def _priors(self, c):
        z = np.zeros_like(c["D"])
        pD = np.log(np.maximum(1.0 - c["D"], 0.02)) if self.prior else z
        if not self.rate_prior:
            pr = z
        elif self.rate_prior == "lognormal":   # mean of a log-normal block-damage model (robust to heavy tails)
            pr = c["lr_mean"] + 0.5 * c["lr_var"]
        elif self.rate_prior == "maxdk":        # mean rainflow rate, bounded below by the Gaussian spectral rate
            pr = np.log(np.maximum(np.maximum(c["D"], c["D_dk"]) / c["tt"], 1e-12))
        else:                                   # arithmetic mean rate D_hat / t
            pr = np.log(np.maximum(c["D"] / c["tt"], 1e-12))
        # standard error of the log mean rate (one pseudo-observation of unit variance avoids se=0 at t=1)
        se = np.sqrt((c["lr_var"] + 1.0) / c["tt"]) if self.shrink else np.ones_like(z)
        return pD, pr, se

    def _inp(self, X):
        return dimensionless(X) if self.sf else X

    def fit(self, tr, va, seed=0):
        Xtr, c, _ = tr
        Xtr = self._inp(Xtr)
        self.sc = Scaler().fit(Xtr)
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        X, w = T(self.sc(Xtr)), T(c["w"] / c["w"].mean())
        pD, pr, se = map(T, self._priors(c))
        yD, yr, y = T(np.log(c["dD"])), T(np.log(c["rate_future"])), T(np.log(c["rul"]))
        self.cD = 0.0 if self.prior else float(yD.mean())          # centre heads without a physics prior
        self.cr = 0.0 if self.rate_prior else float(yr.mean())
        Xv = T(self.sc(self._inp(va[0]))); pDv, prv, sev = map(T, self._priors(va[1])); yv = T(np.log(va[1]["rul"]))
        self.nets = []
        for m in range(self.M):
            net = mlp(X.shape[1], 4)
            def heads(o, pD_, pr_, se_):
                muD = pD_ + o[:, 0] + self.cD
                if self.shrink:   # rate correction and rate variance shrink with the standard error of the prior
                    return muD, o[:, 2], pr_ + se_ * 3.0 * torch.tanh(o[:, 1] / 3.0), torch.logaddexp(o[:, 3], 2 * torch.log(se_))
                return muD, o[:, 2], pr_ + o[:, 1] + self.cr, o[:, 3]
            def loss(b, net=net):
                muD, lvD, mur, lvr = heads(net(X[b]), pD[b], pr[b], se[b])
                mu, lv = muD - mur, torch.logaddexp(lvD, lvr)
                L = gnll(y[b], mu, lv)
                if self.aux:
                    L = L + self.lam * (gnll(yD[b], muD, lvD) + gnll(yr[b], mur, lvr))
                return (w[b] * L).mean()
            def vfn(net=net):
                muD, lvD, mur, lvr = heads(net(Xv), pDv, prv, sev)
                return float(gnll(yv, muD - mur, torch.logaddexp(lvD, lvr)).mean())
            self.nets.append(train_loop(net, loss, len(y), self.epochs, seed * 10 + m, val_fn=vfn))
        self._heads = heads
        return self

    def components(self, X, c):
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        Xt = T(self.sc(self._inp(X))); pD, pr, se = map(T, self._priors(c))
        out = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                out.append([v.numpy() for v in self._heads(net(Xt), pD, pr, se)])
        return out   # list of (muD, lvD, mur, lvr)

    def predict(self, te, rng):
        comps = self.components(te[0], te[1])
        mu = np.stack([a[0] - a[2] for a in comps], 1)
        sd = np.stack([np.exp(0.5 * np.logaddexp(a[1], a[3])) for a in comps], 1)
        return np.exp(mu.mean(1)), mixture_samples(mu, sd, rng)


# ----------------------------------------------------------------------------------------------
class LSTMEnsemble:
    """Black-box deep sequence model: ensemble of LSTMs over the per-block measurement sequence."""
    name = "Deep ensemble LSTM"

    def __init__(self, M=5, epochs=40, h=32):
        self.M, self.epochs, self.h = M, epochs, h

    def _pad(self, data):
        L = max(len(a["seq"]) for _, a in data)
        F = data[0][1]["seq"].shape[1]
        X = np.zeros((len(data), L, F), np.float32); Y = np.zeros((len(data), L), np.float32)
        Mk = np.zeros((len(data), L), np.float32)
        for i, (_, a) in enumerate(data):
            n = len(a["seq"]); X[i, :n] = a["seq"]; v = a["valid"]
            Y[i, :n][v] = np.log(a["rul"][v]); Mk[i, :n] = v / max(v.sum(), 1)
        return X, Y, Mk

    def fit(self, tr_data, va_data, seed=0):
        X, Y, Mk = self._pad(tr_data)
        flat = np.concatenate([a["seq"] for _, a in tr_data])
        self.m, self.s = flat.mean(0), flat.std(0) + 1e-8
        X = ((X - self.m) / self.s).astype(np.float32)
        Xv, Yv, Mv = self._pad(va_data); Xv = ((Xv - self.m) / self.s).astype(np.float32)
        T = lambda a: torch.tensor(a)
        X, Y, Mk, Xv, Yv, Mv = map(T, (X, Y, Mk, Xv, Yv, Mv))
        self.ym = float((Y * Mk).sum() / Mk.sum())
        self.nets = []
        for m in range(self.M):
            torch.manual_seed(seed * 10 + m)
            net = nn.ModuleDict(dict(rnn=nn.LSTM(X.shape[2], self.h, batch_first=True),
                                     head=nn.Linear(self.h, 2)))
            def fwd(x, net=net):
                o, _ = net["rnn"](x); return net["head"](o)
            def loss(b, fwd=fwd):
                o = fwd(X[b]); return (Mk[b] * gnll(Y[b], o[..., 0] + self.ym, o[..., 1])).sum() / len(b)
            def vfn(fwd=fwd):
                o = fwd(Xv); return float((Mv * gnll(Yv, o[..., 0] + self.ym, o[..., 1])).sum() / len(Xv))
            train_loop(net, loss, len(X), self.epochs, seed * 10 + m, bs=32, lr=3e-3, val_fn=vfn)
            self.nets.append(fwd)
        return self

    def predict_data(self, te_data, rng):
        X, _, _ = self._pad(te_data); X = torch.tensor(((X - self.m) / self.s).astype(np.float32))
        mus, sds = [], []
        with torch.no_grad():
            for f in self.nets:
                o = f(X).numpy(); mus.append(o[..., 0] + self.ym); sds.append(np.exp(0.5 * o[..., 1]))
        mu, sd = np.stack(mus, -1), np.stack(sds, -1)
        sel_mu, sel_sd = [], []
        for i, (_, a) in enumerate(te_data):
            n = len(a["seq"]); v = a["valid"]
            sel_mu.append(mu[i, :n][v]); sel_sd.append(sd[i, :n][v])
        mu, sd = np.concatenate(sel_mu), np.concatenate(sel_sd)
        return np.exp(mu.mean(1)), mixture_samples(mu, sd, rng)


# ----------------------------------------------------------------------------------------------
def conformalize(point_va, samp_va, rul_va, samp_te, alpha=0.1):
    """Split-conformal rescaling of the predictive spread in log space (normalised residual score)."""
    lmed = np.log(np.median(samp_va, 1)); lsd = np.std(np.log(samp_va), 1) + 1e-9
    score = np.abs(np.log(rul_va) - lmed) / lsd
    n = len(score)
    qhat = np.quantile(score, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    z = norm.ppf(1 - alpha / 2)
    lam = qhat / z
    lmed_t = np.log(np.median(samp_te, 1))[:, None]
    return np.exp(lmed_t + lam * (np.log(samp_te) - lmed_t)), lam
