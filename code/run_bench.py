"""Run all FatigueBench-RT baselines on one scenario.

python run_bench.py <scen> [--ntrain N] [--seed S] [--methods a,b] [--test_variant v] [--tag t]
"""
import argparse, json, pickle, time, numpy as np
import features as F, models as Mo, metrics as Me
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

p = argparse.ArgumentParser()
p.add_argument("scen"); p.add_argument("--ntrain", type=int, default=0); p.add_argument("--seed", type=int, default=0)
p.add_argument("--methods", default="miner,dirlik,farid,mlp,lstm,pibnn")
p.add_argument("--test_variants", default="base"); p.add_argument("--split", default="iid"); p.add_argument("--tag", default=""); p.add_argument("--save_pred", action="store_true")
a = p.parse_args()

D = ROOT + "/data/"
ld = lambda n: pickle.load(open(D + n + ".pkl", "rb"))
tr, va = ld(f"{a.scen}_train"), ld(f"{a.scen}_val")
tvs = a.test_variants.split(",")
tes = {v: ld(f"{a.scen}_test" + ("" if v == "base" else f"_{v}")) for v in tvs}
rng = np.random.default_rng(a.seed)
if a.split == "extrap":     # life-scale extrapolation: learn on short-lived parts, deploy on long-lived ones
    tr = [t for t in tr if t["K_f"] < 100]; va = [t for t in va if t["K_f"] < 100]
    tes = {v: [t for t in T if t["K_f"] > 150] for v, T in tes.items()}
if a.ntrain:
    tr = [tr[i] for i in rng.choice(len(tr), a.ntrain, replace=False)]
    nva = max(10, a.ntrain // 4)       # validation budget scales with the training budget
    va = [va[i] for i in rng.choice(len(va), min(nva, len(va)), replace=False)]
ntr = len(tr)
EP = int(np.clip(60 * np.sqrt(100 / ntr), 20, 120))     # keep the number of optimiser steps comparable
EPL = int(np.clip(40 * np.sqrt(100 / ntr), 15, 80))
ftr, fva = F.build(tr), F.build(va)
ftes = {v: F.build(t) for v, t in tes.items()}
Str, Sva = Mo.stack(ftr), Mo.stack(fva)
Stes = {v: Mo.stack(f) for v, f in ftes.items()}
for S_ in [Str, Sva] + list(Stes.values()):
    S_[1]["dD"] = np.maximum(S_[1]["dD"], 1e-9)
    S_[1]["rate_future"] = np.maximum(S_[1]["rate_future"], 1e-12)

res = {v: {} for v in tvs}; preds = {v: {} for v in tvs}
for m in a.methods.split(","):
    t0 = time.time()
    seq = False
    if m == "miner": M = Mo.MinerExtrapolation(name="Miner-RT (rainflow)").fit(Str, Sva)
    elif m == "dirlik": M = Mo.MinerExtrapolation(name="Dirlik-RT (spectral)", dirlik=True).fit(Str, Sva)
    elif m == "farid": M = Mo.Farid2022().fit(Str, Sva, a.seed)
    elif m == "mlp": M = Mo.DeepEnsembleMLP(epochs=EP).fit(Str, Sva, a.seed)
    elif m == "lstm": M = Mo.LSTMEnsemble(epochs=EPL).fit(ftr, fva, a.seed); seq = True
    elif m == "pibnn": M = Mo.PIBNN(epochs=EP).fit(Str, Sva, a.seed)
    elif m == "pibnn_noprior": M = Mo.PIBNN(epochs=EP, use_physics_prior=False, name="PI-BNN w/o physics priors").fit(Str, Sva, a.seed)
    elif m == "pibnn_noaux": M = Mo.PIBNN(epochs=EP, use_aux=False, name="PI-BNN w/o damage/rate supervision").fit(Str, Sva, a.seed)
    elif m == "pibnn_rateprior": M = Mo.PIBNN(epochs=EP, rate_prior=True, name="PI-BNN + stationary rate prior").fit(Str, Sva, a.seed)
    elif m.startswith("pibnn_lam"): M = Mo.PIBNN(epochs=EP, lam=float(m[9:]), name=f"PI-BNN lam={m[9:]}").fit(Str, Sva, a.seed)
    elif m == "pibnn_fullfeat": M = Mo.PIBNN(epochs=EP, scale_free=False, name="PI-BNN w/o scale-free inputs").fit(Str, Sva, a.seed)
    elif m == "pibnn_noshrink": M = Mo.PIBNN(epochs=EP, shrink=False, name="PI-BNN w/o standard-error shrinkage").fit(Str, Sva, a.seed)
    elif m == "pibnn_single": M = Mo.PIBNN(M=1, epochs=EP, name="PI-BNN single network").fit(Str, Sva, a.seed)
    ttrain = time.time() - t0
    Pv = M.predict_data(fva, rng) if seq else M.predict(Sva, rng)
    Pv = (np.median(Pv[1], 1), Pv[1])
    for v in tvs:
        t1 = time.time()
        P = M.predict_data(ftes[v], rng) if seq else M.predict(Stes[v], rng)
        P = (np.median(P[1], 1), P[1])          # point prediction = predictive median (all methods)
        tinf = (time.time() - t1) / len(P[0])
        Ste = Stes[v]
        r = Me.evaluate(P[0], P[1], Ste[1], Ste[2]); r["train_s"] = ttrain; r["infer_ms_per_pred"] = 1000 * tinf
        res[v][M.name] = r; preds[v][M.name] = P
        if m not in ("miner", "dirlik"):
            cs, lam = Mo.conformalize(Pv[0], Pv[1], Sva[1]["rul"], P[1])
            rc = Me.evaluate(P[0], cs, Ste[1], Ste[2]); rc["lambda"] = float(lam)
            res[v][M.name + " + CP"] = rc; preds[v][M.name + " + CP"] = (P[0], cs)
            rr = rc
        print(f"[{v}] {M.name:36s} NRMSE {r['NRMSE']:.4f} MALE {r['MALE']:.3f} a-l {r['alpha_lambda']:.3f} PICP {r['PICP90']:.3f} MPIW {r['MPIW90']:.3f} CRPS {r['CRPS']:.4f} ECE {r['ECE']:.3f} PH {r['PH']:.3f} ({ttrain:.0f}s)" + (f" | CP: PICP {rr['PICP90']:.3f} MPIW {rr['MPIW90']:.3f} CRPS {rr['CRPS']:.4f}" if m not in ("miner","dirlik") else ""), flush=True)

tag = a.tag or f"{a.scen}_{a.split}_n{ntr}_s{a.seed}"
json.dump(dict(scen=a.scen, split=a.split, ntest={v: len(t) for v, t in tes.items()}, ntrain=ntr, seed=a.seed, epochs=EP, results=res), open(f"{ROOT}/results/{tag}.json", "w"), indent=1)
if a.save_pred:
    for v in tvs:
        with open(f"{ROOT}/results/pred_{tag}_{v}.pkl", "wb") as f:
            pickle.dump(dict(preds={k: (q[0].astype(np.float32), q[1].astype(np.float32)) for k, q in preds[v].items()},
                             c={k: Stes[v][1][k] for k in ["rul", "Kf", "tt", "D", "D_dk"]}, idx=Stes[v][2]), f)
