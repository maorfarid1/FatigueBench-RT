"""Calibrate a load-life (pseudo-stress) Basquin curve per steel on the CA keyhole tests only.

Notch pseudo-stress = c * P [MPa], Basquin exponent b fitted with sf and su fixed at the SAE values.
The VA tests are never used here."""
import json, sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/home/claude/paper/code")
import fatigue_physics as fp
from parse_keyhole import history

MAT = {"M": (917.0, -0.095, 565.0), "R": (1160.0, -0.075, 820.0)}   # Man-Ten, RQC-100 (sf, b, su)
HISTS = {n: history(n) for n in ["bracket", "suspension", "transmission"]}

def fit_ca(ca, m):
    sf, _, su = MAT[m]
    ca = [r for r in ca if r["mat"] == m]
    Pa = np.array([(r["Pmax"] - r["Pmin"]) / 2 for r in ca]); Pm = np.array([(r["Pmax"] + r["Pmin"]) / 2 for r in ca])
    N = np.array([r["Nf_rev"] / 2 for r in ca])
    pred = lambda th: fp.basquin_life(fp.goodman(np.exp(th[0]) * Pa, np.exp(th[0]) * Pm, su), sf, th[1])
    loss = lambda th: np.sum((np.log(pred(th)) - np.log(N)) ** 2)
    best = min((minimize(loss, [np.log(c0), b0], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
                for c0 in [0.005, 0.01, 0.02] for b0 in [-0.1, -0.15, -0.2]), key=lambda r: r.fun)
    resid = np.log(pred(best.x)) - np.log(N)
    return dict(c=float(np.exp(best.x[0])), b=float(best.x[1]), sf=sf, su=su, n=len(N),
                rms_log=float(np.sqrt(np.mean(resid ** 2))), resid_log10=(resid / np.log(10)).tolist())

def load_block(r):
    x = HISTS[r["history"]]
    ext = x.min() if r["Pext"] < 0 else x.max()
    return x * abs(r["Pext"]) / abs(ext)

def miner_block_damage(r, cal):
    k = cal[r["mat"]]
    P = load_block(r)
    a, m, n = fp.rainflow_cycles(np.concatenate([P, P]) * k["c"])     # two passes: closes the repeated block
    return fp.miner_damage(a, m, n, k["sf"], k["b"], k["su"]) / 2.0

if __name__ == "__main__":
    d = json.load(open("keyhole_tests.json"))
    cal = {m: fit_ca(d["ca"], m) for m in "MR"}
    json.dump(cal, open("calib.json", "w"), indent=1)
    for m, k in cal.items(): print(m, {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in k.items() if kk != "resid_log10"})
    rows = []
    for r in d["va"]:
        Lm = 1.0 / miner_block_damage(r, cal)
        rows.append(dict(id=r["id"], censored=r["censored"], K_f=r["K_f"], K_miner=Lm))
        print("%-6s %s Kf %9.1f  Miner %10.1f  obs/pred %.2f" % (r["id"], "C" if r["censored"] else " ", r["K_f"], Lm, r["K_f"] / Lm))
    rt = np.array([np.log10(q["K_f"] / q["K_miner"]) for q in rows if not q["censored"]])
    print("VA failures: log10(obs/Miner) mean %.2f sd %.2f; within factor 3: %.2f; range of life %.1f to %.0f blocks"
          % (rt.mean(), rt.std(), np.mean(np.abs(rt) < np.log10(3)), min(q["K_f"] for q in rows if not q["censored"]), max(q["K_f"] for q in rows if not q["censored"])))
    json.dump(rows, open("miner_va.json", "w"), indent=1)
