"""Causal real-time features and targets for FatigueBench-RT trajectories."""
import numpy as np


def ewma(x, a):
    y = np.empty_like(x)
    s = x[0]
    for i, v in enumerate(x):
        s = v if i == 0 else (1 - a) * s + a * v
        y[i] = s
    return y


def expanding_std(x):
    n = np.arange(1, len(x) + 1)
    c1, c2 = np.cumsum(x), np.cumsum(x ** 2)
    var = np.maximum(c2 / n - (c1 / n) ** 2, 0.0)
    return np.sqrt(var)


def rolling_slope(y, w=20):
    """Least-squares slope of y over the last w points (causal)."""
    out = np.zeros_like(y)
    for i in range(len(y)):
        j = max(0, i - w + 1)
        seg = y[j:i + 1]
        if seg.size >= 3:
            t = np.arange(seg.size, dtype=float)
            out[i] = np.polyfit(t, seg, 1)[0]
    return out


FEATURE_NAMES = [
    "sf", "neg_b", "su", "log_t", "log_D_miner", "log_rate_fast", "log_rate_slow", "log_rate_mean",
    "log_rate_dirlik", "log_rf_over_dirlik", "rms_su", "mean_su", "kurt", "skew", "irr", "fbar",
    "xi90_su", "std_log_rate", "slope_log_rate", "log_one_minus_D", "geo_rate_fast", "geo_rate_cum",
]
PER_BLOCK_NAMES = ["sf", "neg_b", "su", "log_d_rf", "log_d_dk", "rms_su", "mean_su", "kurt", "skew",
                   "irr", "fbar", "xi90_su"]


def traj_features(t):
    """Return (X [K,F], aux dict) for every observed block k=0..K-1 of trajectory t."""
    K = len(t["d_rf"])
    su = t["su"]
    d = np.maximum(t["d_rf"], 1e-12)
    ld = np.log(d)
    D = np.cumsum(d)
    tt = np.arange(1, K + 1, dtype=float)
    r_fast, r_slow = ewma(d, 0.3), ewma(d, 0.05)
    r_dk = ewma(np.maximum(t["d_dk"], 1e-12), 0.05)
    X = np.column_stack([
        np.full(K, t["sf"] / 1000.0), np.full(K, -t["b"] * 10.0), np.full(K, su / 1000.0),
        np.log(tt), np.log(D), np.log(r_fast), np.log(r_slow), np.log(D / tt), np.log(r_dk),
        ewma(ld - np.log(np.maximum(t["d_dk"], 1e-12)), 0.1),
        ewma(t["rms"] / su, 0.1), ewma(t["mean"] / su, 0.1), ewma(t["kurt"], 0.1), ewma(t["skew"], 0.1),
        ewma(t["irr"], 0.1), ewma(t["fbar"], 0.1) / 50.0, t["xi90"] / su,
        expanding_std(ld), rolling_slope(ld), np.log(np.maximum(1.0 - D, 0.02)),
        ewma(ld, 0.1), np.cumsum(ld) / tt,
    ])
    seq = np.column_stack([
        np.full(K, t["sf"] / 1000.0), np.full(K, -t["b"] * 10.0), np.full(K, su / 1000.0), ld,
        np.log(np.maximum(t["d_dk"], 1e-12)), t["rms"] / su, t["mean"] / su, t["kurt"], t["skew"],
        t["irr"], t["fbar"] / 50.0, t["xi90"] / su,
    ])
    # targets
    Kf = t["K_f"]
    kf = int(np.floor(Kf))
    D_crit = (D[kf - 1] if kf >= 1 else 0.0) + (d[kf] if kf < K else d[-1]) * (Kf - kf)
    rul = Kf - tt                       # remaining blocks after observing block k
    valid = rul > 1e-6
    dD = D_crit - D
    rate_future = np.where(valid, dD / np.maximum(rul, 1e-6), np.nan)
    aux = dict(lr_mean=np.cumsum(ld) / tt, lr_var=expanding_std(ld) ** 2,
               D_dk=np.cumsum(np.maximum(t["d_dk"], 1e-12)), rul=rul, valid=valid, dD=dD, rate_future=rate_future, D=D, r_fast=r_fast, r_slow=r_slow,
               r_dk=r_dk, tt=tt, Kf=Kf, D_crit=D_crit, seq=seq,
               # inputs of the Farid (2022) ANN+GPR model: material + cumulative mean + cumulative xi90
               farid=np.column_stack([np.full(K, t["sf"]), np.full(K, t["b"]), np.full(K, su),
                                      np.cumsum(t["mean"]) / tt, np.cumsum(t["xi90"]) / tt]))
    return X, aux


def build(trajs):
    out = []
    for t in trajs:
        X, aux = traj_features(t)
        out.append((X, aux))
    return out


def dimensionless(X):
    """Scale-free (Buckingham-pi style) features: invariant to the absolute damage rate / life scale."""
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    c = lambda n: X[:, i[n]]
    D = np.exp(c("log_D_miner"))
    return np.column_stack([
        D, c("log_D_miner"), c("log_rate_fast") - c("log_rate_mean"), c("log_rate_slow") - c("log_rate_mean"),
        c("geo_rate_fast") - c("geo_rate_cum"), c("log_rate_mean") - c("geo_rate_cum"), c("std_log_rate"),
        c("slope_log_rate") * np.exp(c("log_t")), c("kurt"), c("skew"), c("irr"), c("log_rf_over_dirlik"),
        c("mean_su"), c("neg_b"),
    ])
