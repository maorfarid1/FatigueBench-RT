"""FatigueBench-RT trajectory generator.

Each trajectory is one component monitored block by block (1 block = 1 h of operation, represented by a
4 s measured window) until fatigue failure. Every component has its own usage profile ("mission mix"):
a small library of operating regimes (PSD shape, load level, mean stress, non-Gaussianity) visited in a
random semi-Markov sequence with component-specific probabilities.

Usage: python gen.py <scenario> <split> <n> <seed> [variant]
"""
import sys, pickle, numpy as np
from multiprocessing import Pool
from scipy.stats import skew as _skew, kurtosis as _kurt
import fatigue_physics as fp
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

SCEN = {
    "S1": dict(nonstat=False, nongauss=False, seq=False),   # stationary Gaussian
    "S2": dict(nonstat=True, nongauss=False, seq=False),    # non-stationary (mission mix) Gaussian
    "S3": dict(nonstat=True, nongauss=True, seq=False),     # non-stationary non-Gaussian
    "S4": dict(nonstat=True, nongauss=True, seq=True),      # + load drift and load-sequence effects (DCA)
}
SAE_MATERIALS = [(917.0, -0.095, 565.0), (1160.0, -0.075, 820.0)]  # Man-Ten, RQC-100 (SAE program)
EPS_SIG = 0.25     # log-std of unobserved material life scatter
NOISE = 0.02       # sensor noise level relative to load rms
MAX_BLOCKS = 1500
N_CAL = 4          # calibration windows per regime type


def new_regime(cfg, rng, kurt_range, su):
    npk = rng.integers(1, 3)
    peaks = []
    for _ in range(npk):
        fc = rng.uniform(5, 80)
        peaks.append((fc, rng.uniform(1.0, max(1.5, 0.3 * fc)), rng.uniform(0.2, 1.0)))
    reg = dict(peaks=peaks, rho=1.0, mfrac=rng.uniform(0.0, 0.25), skew=0.0, kurt=3.0, mod=0.0)
    if cfg["nonstat"]:
        reg["rho"] = float(np.exp(rng.uniform(np.log(0.75), np.log(1.35))))
        reg["mod"] = rng.uniform(0.0, 0.4)
    if cfg["nongauss"]:
        reg["skew"] = rng.uniform(-0.4, 0.4)
        reg["kurt"] = rng.uniform(*kurt_range)
    return reg


def unit_window(reg, rng):
    u = fp.synth_gaussian(reg["peaks"], rng)
    x = fp.hermite_transform(u, reg["skew"], reg["kurt"]) if reg["kurt"] > 3.0 else u
    if reg["mod"] > 0:
        t = np.arange(fp.NW) / fp.FS
        x = x * (1.0 + reg["mod"] * np.sin(2 * np.pi * t / fp.T_W + rng.uniform(0, 2 * np.pi)))
    return x


def simulate(args):
    scen, seed, variant = args
    for rep in range(20):
        out = _simulate(scen, seed * 31 + rep, variant)
        if out["K_f"] is not None and out["K_f"] >= 15:
            return out
    return out


def _simulate(scen, seed, variant):
    cfg = SCEN[scen]
    rng = np.random.default_rng(seed)
    kurt_range = (7.0, 10.0) if variant == "ood_kurt" else (3.0, 7.0)
    for attempt in range(50):
        if variant == "sae":
            sf, b, su = SAE_MATERIALS[rng.integers(0, 2)]
        else:
            sf, b, su = rng.uniform(900, 1600), rng.uniform(-0.15, -0.07), rng.uniform(500, 1200)
        eps = float(np.exp(rng.normal(0, EPS_SIG)))
        L_target = float(np.exp(rng.uniform(np.log(40), np.log(250))))
        ntype = int(rng.integers(2, 5)) if cfg["nonstat"] else 1
        regs = [new_regime(cfg, rng, kurt_range, su) for _ in range(ntype)]
        p_use = rng.dirichlet(np.ones(ntype))
        cal = []
        for j, r in enumerate(regs):
            for _ in range(N_CAL):
                xu = unit_window(r, rng)
                a, m, c = fp.rainflow_cycles(xu)
                cal.append((j, r, a, m, c, np.abs(xu).max()))
        w = np.array([p_use[j] for j, *_ in cal])

        def rate(s):
            d = [fp.miner_damage(s * r["rho"] * a, r["mfrac"] * su + s * r["rho"] * m, c, sf, b, su)
                 for _, r, a, m, c, _ in cal]
            return np.sum(w * np.array(d)) / w.sum() * fp.SCALE

        lo, hi = np.log(1.0), np.log(su)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if rate(np.exp(mid)) * L_target > 1.0:
                hi = mid
            else:
                lo = mid
        s = float(np.exp(0.5 * (lo + hi)))
        peak = max(r["mfrac"] * su + s * r["rho"] * mx for _, r, _, _, _, mx in cal)
        if peak < 0.9 * su:
            break
    # reference load-level life for the damage-curve approach (true material, usage-weighted)
    lneq = []
    for _, r, a, m, c, _ in cal:
        d = fp.miner_damage(s * r["rho"] * a, r["mfrac"] * su + s * r["rho"] * m, c, sf, b, su) / eps
        lneq.append(np.log(c.sum() / max(d, 1e-300)))
    lneq_ref = float(np.sum(w * np.array(lneq)) / w.sum())
    f_end = float(np.exp(rng.uniform(np.log(0.75), np.log(1.35)))) if cfg["seq"] else 1.0

    rec = {k: [] for k in ["rms", "mean", "skew", "kurt", "irr", "fbar", "xi90", "d_rf", "d_dk",
                           "d_true", "ncyc", "regime"]}
    D, k, K_f, lneq_s = 0.0, 0, None, 0.0
    ri, left = int(rng.choice(ntype, p=p_use)), int(rng.integers(1, 9))
    while k < MAX_BLOCKS:
        if left <= 0:
            ri, left = int(rng.choice(ntype, p=p_use)), int(rng.integers(1, 9))
        r = regs[ri]
        xu = unit_window(r, rng)
        drift = 1.0 + (f_end - 1.0) * (1.0 - np.exp(-k / (0.3 * L_target))) if cfg["seq"] else 1.0
        amp = s * r["rho"] * drift
        xc = r["mfrac"] * su + amp * xu
        xm = xc + rng.normal(0, NOISE * amp, xc.size)
        a, m, c = fp.rainflow_cycles(xc)
        d_true = fp.miner_damage(a, m, c, sf, b, su) / eps * fp.SCALE
        am, mm, cm = fp.rainflow_cycles(xm)
        d_rf = fp.miner_damage(am, mm, cm, sf, b, su) * fp.SCALE
        _, _, mom = fp.spectral_moments(xm)
        d_dk = fp.dirlik_damage_rate(mom, xm.mean(), sf, b, su) * fp.T_B
        xd = xm - xm.mean()
        pk = xd[1:-1][(xd[1:-1] > xd[:-2]) & (xd[1:-1] > xd[2:])]
        rec["rms"].append(xd.std()); rec["mean"].append(xm.mean())
        rec["skew"].append(_skew(xd)); rec["kurt"].append(_kurt(xd, fisher=False))
        rec["irr"].append(mom[2] / np.sqrt(mom[0] * mom[3])); rec["fbar"].append(mom[1] / mom[0])
        rec["xi90"].append(np.percentile(np.abs(pk), 90) if pk.size else 0.0)
        rec["d_rf"].append(d_rf); rec["d_dk"].append(d_dk); rec["d_true"].append(d_true)
        rec["ncyc"].append(c.sum() * fp.SCALE); rec["regime"].append(ri)
        if cfg["seq"]:   # damage-curve approach (Manson-Halford) on the smoothed load-level life
            ln = np.log(c.sum() * fp.SCALE / max(d_true, 1e-300))
            lneq_s = ln if k == 0 else 0.8 * lneq_s + 0.2 * ln
            q = float(np.clip(np.exp(0.4 * (lneq_s - lneq_ref)), 0.5, 2.0))
            r_eq = D ** (1.0 / q)
            if r_eq + d_true >= 1.0:
                K_f = k + (1.0 - r_eq) / d_true
            else:
                D = (r_eq + d_true) ** q
        else:
            if D + d_true >= 1.0:
                K_f = k + (1.0 - D) / d_true
            else:
                D += d_true
        k += 1
        left -= 1
        if K_f is not None:
            break
    out = {kk: np.asarray(v, np.float64) for kk, v in rec.items()}
    out.update(sf=sf, b=b, su=su, eps=eps, K_f=K_f, s=s, L_target=L_target, seed=seed, scen=scen,
               variant=variant, ntype=ntype, p_use=p_use, f_end=f_end)
    return out


if __name__ == "__main__":
    scen, split, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    variant = sys.argv[5] if len(sys.argv) > 5 else "base"
    seeds = [seed * 100000 + i for i in range(n)]
    with Pool(2) as p:
        trajs = p.map(simulate, [(scen, sd, variant) for sd in seeds], chunksize=4)
    trajs = [t for t in trajs if t["K_f"] is not None]
    tag = f"{scen}_{split}" + ("" if variant == "base" else f"_{variant}")
    with open(f"{ROOT}/data/{tag}.pkl", "wb") as f:
        pickle.dump(trajs, f)
    life = np.array([t["K_f"] for t in trajs])
    print(tag, len(trajs), "life blocks: median %.1f  [%.1f, %.1f]" % (np.median(life), life.min(), life.max()), flush=True)
