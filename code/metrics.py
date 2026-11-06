"""Prognostic metrics: accuracy, calibration, sharpness, CRPS, alpha-lambda and prognostic horizon."""
import numpy as np

ALPHA_PI = 0.10          # 90% prediction intervals


def crps_samples(s, y):
    """CRPS from samples (energy form), per point. s [n,S], y [n]."""
    s = np.sort(s, 1)
    S = s.shape[1]
    t1 = np.mean(np.abs(s - y[:, None]), 1)
    # E|X-X'| via sorted-sample identity
    i = np.arange(1, S + 1)
    t2 = 2.0 * np.sum(s * (2 * i - S - 1), 1) / (S * S)
    return t1 - 0.5 * t2


def evaluate(point, samples, c, idx, frac_lo=0.1, frac_hi=0.95):
    rul, Kf = c["rul"], c["Kf"]
    frac = c["tt"] / Kf
    sel = (frac >= frac_lo) & (frac <= frac_hi)
    lo, hi = np.quantile(samples, [ALPHA_PI / 2, 1 - ALPHA_PI / 2], axis=1)
    err = (point - rul) / Kf
    out = dict(
        NRMSE=float(np.sqrt(np.mean(err[sel] ** 2))),
        MALE=float(np.mean(np.abs(np.log(point[sel]) - np.log(rul[sel])))),
        alpha_lambda=float(np.mean(np.abs(point[sel] - rul[sel]) <= 0.2 * rul[sel])),
        PICP90=float(np.mean((rul[sel] >= lo[sel]) & (rul[sel] <= hi[sel]))),
        MPIW90=float(np.mean(np.log(hi[sel] / lo[sel]))),                       # width of log-interval
        MPIW_lin=float(np.median((hi[sel] - lo[sel]) / Kf[sel])),
        CRPS=float(np.mean(crps_samples(np.log(samples[sel]), np.log(rul[sel])))),  # CRPS of log RUL
        CRPS_lin=float(np.median(crps_samples(samples[sel], rul[sel]) / Kf[sel])),
    )
    # calibration error over central intervals 10..90 %
    levels = np.linspace(0.1, 0.9, 9)
    cov = []
    for a in levels:
        l, h = np.quantile(samples[sel], [(1 - a) / 2, (1 + a) / 2], axis=1)
        cov.append(np.mean((rul[sel] >= l) & (rul[sel] <= h)))
    out["ECE"] = float(np.mean(np.abs(np.array(cov) - levels)))
    out["_cov"] = (levels.tolist(), [float(v) for v in cov])
    # prognostic horizon (Saxena et al.): fraction of life before EOL from which the point prediction
    # stays within +-10% of total life until EOL
    ph = []
    for j in np.unique(idx):
        m = idx == j
        ok = np.abs(err[m]) <= 0.10
        bad = np.where(~ok)[0]
        first = 0 if bad.size == 0 else bad[-1] + 1
        ph.append(rul[m][first] / Kf[m][0] if first < m.sum() else 0.0)
    out["PH"] = float(np.mean(ph))
    return out


def by_fraction(point, samples, c, bins=np.linspace(0.05, 0.95, 10)):
    rul, Kf = c["rul"], c["Kf"]
    frac = c["tt"] / Kf
    lo, hi = np.quantile(samples, [0.05, 0.95], axis=1)
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (frac >= a) & (frac < b)
        if m.sum() == 0:
            continue
        rows.append(dict(frac=float((a + b) / 2), NRMSE=float(np.sqrt(np.mean(((point[m] - rul[m]) / Kf[m]) ** 2))),
                         PICP90=float(np.mean((rul[m] >= lo[m]) & (rul[m] <= hi[m]))),
                         MPIW90=float(np.mean((hi[m] - lo[m]) / Kf[m]))))
    return rows
