"""Sim-to-real test: train every method on simulated FatigueBench-RT data, test on the SAE keyhole
variable-amplitude run-to-failure tests (no real data used in training or calibration of the ML models).

python run_real.py --train S4 --seed 0
"""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
for _d in ("tex/figs", "figs", "results", "data"): _os.makedirs(_os.path.join(ROOT, _d), exist_ok=True)
import argparse, json, pickle, time, numpy as np
import features as F, models as Mo, metrics as Me

p = argparse.ArgumentParser()
p.add_argument("--train", default="S4"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--methods", default="miner,dirlik,farid,mlp,lstm,pibnn")
a = p.parse_args()
D = ROOT + "/data/"
ld = lambda n: pickle.load(open(D + n + ".pkl", "rb"))
tr, va, te = ld(f"{a.train}_train"), ld(f"{a.train}_val"), ld("REAL_keyhole")
rng = np.random.default_rng(a.seed)
EP, EPL = int(np.clip(60 * np.sqrt(100 / len(tr)), 20, 120)), int(np.clip(40 * np.sqrt(100 / len(tr)), 15, 80))
ftr, fva, fte = F.build(tr), F.build(va), F.build(te)
Str, Sva, Ste = Mo.stack(ftr), Mo.stack(fva), Mo.stack(fte)
for S_ in (Str, Sva, Ste):
    S_[1]["dD"] = np.maximum(S_[1]["dD"], 1e-9); S_[1]["rate_future"] = np.maximum(S_[1]["rate_future"], 1e-12)

# evaluation points: every observed block for short tests, 19 points at 5..95% of life otherwise,
# so that each specimen carries a similar weight whatever its life
c, idx = Ste[1], Ste[2]
keep = np.zeros(len(idx), bool)
for j in np.unique(idx):
    m = np.where(idx == j)[0]
    if len(m) <= 40:
        keep[m] = True
    else:
        tt, Kf = c["tt"][m], c["Kf"][m][0]
        for f in np.arange(0.05, 0.951, 0.05):
            keep[m[np.argmin(np.abs(tt - f * Kf))]] = True
sub = lambda S: (S[0][keep], {k: v[keep] for k, v in S[1].items()}, S[2][keep])
Ste_s = sub(Ste)

res, preds = {}, {}
for m in a.methods.split(","):
    t0 = time.time(); seq = False
    if m == "miner": M = Mo.MinerExtrapolation(name="Miner-RT (rainflow)").fit(Str, Sva)
    elif m == "dirlik": M = Mo.MinerExtrapolation(name="Dirlik-RT (spectral)", dirlik=True).fit(Str, Sva)
    elif m == "farid": M = Mo.Farid2022().fit(Str, Sva, a.seed)
    elif m == "mlp": M = Mo.DeepEnsembleMLP(epochs=EP).fit(Str, Sva, a.seed)
    elif m == "lstm": M = Mo.LSTMEnsemble(epochs=EPL).fit(ftr, fva, a.seed); seq = True
    elif m == "pibnn": M = Mo.PIBNN(epochs=EP).fit(Str, Sva, a.seed)
    Pv = M.predict_data(fva, rng) if seq else M.predict(Sva, rng)
    Pv = (np.median(Pv[1], 1), Pv[1])
    if seq:
        P = M.predict_data(fte, rng); P = (P[0][keep], P[1][keep])
    else:
        P = M.predict(Ste_s, rng)
    P = (np.median(P[1], 1), P[1])
    r = Me.evaluate(P[0], P[1], Ste_s[1], Ste_s[2]); res[M.name] = r; preds[M.name] = P
    line = f"{M.name:36s} MALE {r['MALE']:.3f} a-l {r['alpha_lambda']:.3f} PICP {r['PICP90']:.3f} MPIW {r['MPIW90']:.3f} CRPS {r['CRPS']:.4f} ECE {r['ECE']:.3f}"
    if m not in ("miner", "dirlik"):
        cs, lam = Mo.conformalize(Pv[0], Pv[1], Sva[1]["rul"], P[1])
        rc = Me.evaluate(P[0], cs, Ste_s[1], Ste_s[2]); rc["lambda"] = float(lam)
        res[M.name + " + CP"] = rc; preds[M.name + " + CP"] = (P[0], cs)
        line += f" | CP: PICP {rc['PICP90']:.3f} MPIW {rc['MPIW90']:.3f} CRPS {rc['CRPS']:.4f}"
    print(line + f" ({time.time() - t0:.0f}s)", flush=True)

tag = f"REAL_{a.train}_s{a.seed}"
json.dump(dict(train=a.train, seed=a.seed, ntest=len(te), npts=int(keep.sum()), results=res),
          open(f"{ROOT}/results/{tag}.json", "w"), indent=1)
with open(f"{ROOT}/results/pred_{tag}.pkl", "wb") as f:
    pickle.dump(dict(preds={k: (q[0].astype(np.float32), q[1].astype(np.float32)) for k, q in preds.items()},
                     c={k: Ste_s[1][k] for k in ["rul", "Kf", "tt", "D"]}, idx=Ste_s[2],
                     ids=[t["id"] for t in te]), f)
