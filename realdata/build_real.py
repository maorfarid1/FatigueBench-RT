"""Turn the SAE keyhole variable-amplitude tests into FatigueBench-RT trajectories (real-data test set).

One monitoring block = one pass of the measured service history (bracket 5936, suspension 2506,
transmission 1708 reversals), scaled to the specimen's load level. The notch pseudo-stress uses the
load-life curve calibrated on the constant-amplitude keyhole tests only (calib_keyhole.py). The
peak-valley sequence is turned into a signal by half-cosine interpolation (8 samples per reversal),
so the spectral (Dirlik) features are defined. Sensor noise and block statistics follow gen.py.
Failure time K_f is the measured number of blocks at the end of the test (final fracture).
"""
import json, pickle, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import skew as _skew, kurtosis as _kurt
sys.path.insert(0, "/home/claude/paper/code")
import fatigue_physics as fp
from parse_keyhole import history

NS = 8          # samples per reversal
R_NOISE = 12    # noisy realisations per specimen (sampled block by block)
NOISE = 0.02    # as in gen.py
CAL = json.load(open("/home/claude/paper/realdata/calib.json"))
HISTS = {n: history(n) for n in ["bracket", "suspension", "transmission"]}


def to_signal(tp):
    """Half-cosine interpolation between successive turning points."""
    a, b = tp[:-1, None], tp[1:, None]
    w = 0.5 * (1 - np.cos(np.pi * np.arange(NS) / NS))[None, :]
    return np.concatenate([(a + (b - a) * w).ravel(), tp[-1:]])


def block_stats(x, k, rng):
    """Same per-block measurements as gen.py, for one noisy realisation of the block signal x."""
    xm = x + rng.normal(0, NOISE * x.std(), x.size)
    am, mm, cm = fp.rainflow_cycles(xm)
    d_rf = fp.miner_damage(am, mm, cm, k["sf"], k["b"], k["su"])
    _, _, mom = fp.spectral_moments(xm)
    d_dk = fp.dirlik_damage_rate(mom, xm.mean(), k["sf"], k["b"], k["su"]) * (x.size / fp.FS)
    xd = xm - xm.mean()
    pk = xd[1:-1][(xd[1:-1] > xd[:-2]) & (xd[1:-1] > xd[2:])]
    return dict(rms=xd.std(), mean=xm.mean(), skew=_skew(xd), kurt=_kurt(xd, fisher=False),
                irr=mom[2] / np.sqrt(mom[0] * mom[3]), fbar=mom[1] / mom[0],
                xi90=np.percentile(np.abs(pk), 90) if pk.size else 0.0, d_rf=d_rf, d_dk=d_dk)


def build(r):
    k = CAL[r["mat"]]
    x = HISTS[r["history"]]
    ext = x.min() if r["Pext"] < 0 else x.max()
    P = x * abs(r["Pext"]) / abs(ext)
    sig = to_signal(P * k["c"])
    rng = np.random.default_rng(abs(hash(r["id"])) % 2**32)
    reals = [block_stats(sig, k, rng) for _ in range(R_NOISE)]
    K = int(np.floor(r["K_f"])) + 1
    pick = rng.integers(0, R_NOISE, K)
    out = {f: np.array([reals[j][f] for j in pick], np.float64) for f in reals[0]}
    a, m, n = fp.rainflow_cycles(sig)
    out.update(d_true=out["d_rf"].copy(), ncyc=np.full(K, n.sum()), regime=np.zeros(K),
               sf=k["sf"], b=k["b"], su=k["su"], eps=1.0, K_f=r["K_f"], s=k["c"], L_target=r["K_f"],
               seed=0, scen="SAE-keyhole", variant="real", ntype=1, p_use=np.ones(1), f_end=1.0,
               id=r["id"], history=r["history"], mat=r["mat"], level=r["level"], Pext=r["Pext"])
    return out


if __name__ == "__main__":
    d = json.load(open("/home/claude/paper/realdata/keyhole_tests.json"))
    va = [r for r in d["va"] if not r["censored"]]
    with Pool(2) as p:
        trajs = p.map(build, va, chunksize=1)
    pickle.dump(trajs, open("/home/claude/paper/data/REAL_keyhole.pkl", "wb"))
    for t in trajs:
        print(t["id"], "K_f %.1f" % t["K_f"], "blocks", len(t["d_rf"]), "d/block %.3g" % np.median(t["d_rf"]),
              "Miner life %.1f" % (1 / np.median(t["d_rf"])), "kurt %.1f" % np.median(t["kurt"]))
