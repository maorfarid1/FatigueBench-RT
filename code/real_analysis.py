"""Summaries for the SAE keyhole real-data test: seed-averaged metrics, leave-one-group-out conformal
recalibration on the real tests, and a replicate-scatter reference."""
import json, pickle, numpy as np
from scipy.stats import norm
import metrics as Me

R = "/home/claude/paper/results/"
SEEDS = [0, 1, 2]


def groups(ids):
    return np.array([i[0] + i[1] for i in ids])          # history x steel, e.g. "SM", "BR"


def cross_conformal(pt, s, rul, gidx, alpha=0.1):
    """Leave-one-group-out split conformal in log space: the spread of each group is rescaled with
    the quantile of normalised residuals of the other groups (no information from the group itself)."""
    lmed = np.log(np.median(s, 1)); lsd = np.std(np.log(s), 1) + 1e-9
    score = np.abs(np.log(rul) - lmed) / lsd
    out = np.empty_like(s); lams = {}
    z = norm.ppf(1 - alpha / 2)
    for g in np.unique(gidx):
        cal = gidx != g
        n = cal.sum()
        q = np.quantile(score[cal], min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
        lam = q / z; lams[g] = float(lam)
        m = gidx == g
        out[m] = np.exp(lmed[m, None] + lam * (np.log(s[m]) - lmed[m, None]))
    return out, lams


def main():
    rows = {}
    for sd in SEEDS:
        d = pickle.load(open(R + f"pred_REAL_S4_s{sd}.pkl", "rb"))
        c, idx = d["c"], d["idx"]
        gid = groups(d["ids"])[idx]
        for name, (pt, s) in d["preds"].items():
            pt = pt.astype(float); s = s.astype(float)
            rows.setdefault(name, []).append(Me.evaluate(pt, s, c, idx))
            if not name.endswith("+ CP"):
                cs, lams = cross_conformal(pt, s, c["rul"], gid)
                r = Me.evaluate(pt, cs, c, idx); r["lambda_mean"] = float(np.mean(list(lams.values())))
                rows.setdefault(name + " + CP(real)", []).append(r)
    keys = ["MALE", "alpha_lambda", "PICP90", "MPIW90", "CRPS", "ECE", "NRMSE"]
    summ = {n: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))] for k in keys}
            for n, v in rows.items()}
    for n, v in rows.items():
        if "lambda_mean" in v[0]:
            summ[n]["lambda"] = float(np.mean([r["lambda_mean"] for r in v]))
    # total-life accuracy of the CA-calibrated Miner model and replicate scatter
    mv = json.load(open("/home/claude/paper/realdata/miner_va.json"))
    va = json.load(open("/home/claude/paper/realdata/keyhole_tests.json"))["va"]
    fail = [q for q in mv if not q["censored"]]
    lr = np.log10([q["K_f"] / q["K_miner"] for q in fail])
    lev = {}
    for r in va:
        if not r["censored"]:
            lev.setdefault((r["history"], r["mat"], r["level"]), []).append(np.log10(r["K_f"]))
    rep = [x - np.mean(v) for v in lev.values() if len(v) > 1 for x in v]
    nrep = [len(v) for v in lev.values() if len(v) > 1]
    life = dict(n_fail=len(fail), n_ca=len(json.load(open("/home/claude/paper/realdata/keyhole_tests.json"))["ca"]),
                bias_log10=float(lr.mean()), sd_log10=float(lr.std(ddof=1)), within3=float(np.mean(np.abs(lr) < np.log10(3))),
                within2=float(np.mean(np.abs(lr) < np.log10(2))),
                rep_sd_log10=float(np.sqrt(np.sum(np.square(rep)) / (len(rep) - len(nrep)))),
                life_min=float(min(q["K_f"] for q in fail)), life_max=float(max(q["K_f"] for q in fail)))
    cal = json.load(open("/home/claude/paper/realdata/calib.json"))
    life["calib"] = {m: dict(c=v["c"], b=v["b"], rms_log10=v["rms_log"] / np.log(10), n=v["n"]) for m, v in cal.items()}
    json.dump(dict(methods=summ, life=life), open(R + "real_summary.json", "w"), indent=1)
    for n, v in summ.items():
        print(f"{n:40s} " + " ".join(f"{k} {v[k][0]:.3f}" for k in keys) + (f" lam {v['lambda']:.2f}" if "lambda" in v else ""))
    print(life)


if __name__ == "__main__":
    main()
