"""Uncertainty decomposition and time-scale equivariance test (Fig. 9)."""
import pickle, copy, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import features as F, models as Mo
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False})
D = ROOT + "/data/"
ld = lambda n: pickle.load(open(D + n + ".pkl", "rb"))
tr, va, te = ld("S3_train"), ld("S3_val"), ld("S3_test")


def prep(x):
    S = Mo.stack(F.build(x))
    S[1]["dD"] = np.maximum(S[1]["dD"], 1e-9); S[1]["rate_future"] = np.maximum(S[1]["rate_future"], 1e-12)
    return S


Str, Sva, Ste = prep(tr), prep(va), prep(te)
EP = int(np.clip(60 * np.sqrt(100 / len(tr)), 20, 120))
pib = Mo.PIBNN(epochs=EP).fit(Str, Sva, 0)
mlp = Mo.DeepEnsembleMLP(epochs=EP).fit(Str, Sva, 0)

# --- (a) uncertainty decomposition on S3 test
comps = pib.components(Ste[0], Ste[1])
muD = np.stack([c[0] for c in comps], 1); vD = np.stack([np.exp(c[1]) for c in comps], 1)
mur = np.stack([c[2] for c in comps], 1); vr = np.stack([np.exp(c[3]) for c in comps], 1)
alea_D, alea_r = vD.mean(1), vr.mean(1)
epi = (muD - mur).var(1)
frac = Ste[1]["tt"] / Ste[1]["Kf"]
bins = np.linspace(0.05, 0.95, 10)
xs, aD, ar, ep = [], [], [], []
for a, b in zip(bins[:-1], bins[1:]):
    s = (frac >= a) & (frac < b)
    xs.append((a + b) / 2); aD.append(np.median(alea_D[s])); ar.append(np.median(alea_r[s])); ep.append(np.median(epi[s]))

# --- (b) time-scale equivariance: dilate a test component by c (same damage, c times slower)


def dilate(t, c):
    u = copy.deepcopy(t)
    for k in ["rms", "mean", "skew", "kurt", "irr", "fbar", "xi90", "ncyc", "regime"]:
        u[k] = np.repeat(t[k], c)
    for k in ["d_rf", "d_dk", "d_true"]:
        u[k] = np.repeat(t[k], c) / c
    u["K_f"] = t["K_f"] * c
    return u


lives = np.array([t["K_f"] for t in te])
cand = [te[i] for i in np.argsort(np.abs(lives - 60))[:40]]
ratios = {"PI-BNN (proposed)": [], "Deep ensemble MLP": []}
curves = {}
for c in [1, 2, 4, 8]:
    S = prep([dilate(t, c) for t in cand])
    fr = S[1]["tt"] / S[1]["Kf"]
    for name, M in [("PI-BNN (proposed)", pib), ("Deep ensemble MLP", mlp)]:
        pt, smp = M.predict(S, np.random.default_rng(0))
        pt = np.median(smp, 1)
        rel = pt / S[1]["Kf"]
        sel = (fr >= 0.1) & (fr <= 0.95)
        male = np.mean(np.abs(np.log(pt[sel]) - np.log(S[1]["rul"][sel])))
        ratios[name].append((c, float(male)))
        # curve for first candidate
        for j0 in range(len(cand)):
            m0 = S[2] == j0
            curves[(name, c, j0)] = (fr[m0], rel[m0], (S[1]["rul"] / S[1]["Kf"])[m0])

# representative component: median PI-BNN error at c=1
errs = [np.mean(np.abs(np.log(np.maximum(curves[("PI-BNN (proposed)", 1, j)][1], 1e-6)) - np.log(np.maximum(curves[("PI-BNN (proposed)", 1, j)][2], 1e-6)))) for j in range(len(cand))]
jsel = int(np.argsort(np.abs(np.array(errs) - np.median(errs)))[0])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
axs[0].plot(xs, aD, "o-", ms=2.5, color="#e9a03b", label="aleatoric, damage state $\\sigma_D^2$")
axs[0].plot(xs, ar, "s-", ms=2.5, color="#2a9d8f", label="aleatoric, load forecast $\\sigma_r^2$")
axs[0].plot(xs, ep, "^-", ms=2.5, color="#6c3483", label="epistemic (ensemble)")
axs[0].set_yscale("log"); axs[0].set_xlabel("life fraction $t/K_f$"); axs[0].set_ylabel("variance of $\\log$ RUL (median)")
axs[0].legend(fontsize=5.8); axs[0].set_title("(a) uncertainty sources, S3")
for j, name in enumerate(["PI-BNN (proposed)", "Deep ensemble MLP"]):
    ax = axs[1 + j]
    for c, col in zip([1, 2, 4, 8], ["#000000", "#1f77b4", "#2ca02c", "#d62728"]):
        fr, rel, true = curves[(name, c, jsel)]
        ax.plot(fr, rel, color=col, lw=0.9, label=f"dilation $c$={c}")
    ax.plot(fr, true, "k--", lw=0.6, label="true")
    ax.set_xlabel("life fraction $t/K_f$"); ax.set_ylabel("predicted RUL$/K_f$"); ax.set_ylim(0, 2.0)
    ax.set_title("(b) PI-BNN, time dilation" if j == 0 else "(c) deep ens. MLP, time dilation")
    if j == 0: ax.legend(fontsize=5.8)
plt.tight_layout(); plt.savefig(ROOT + "/figs/fig9_uq_equivariance.pdf"); plt.savefig(ROOT + "/figs/fig9_uq_equivariance.png", dpi=200)
json.dump(dict(decomp=dict(frac=[float(v) for v in xs], aleaD=[float(v) for v in aD], alear=[float(v) for v in ar], epi=[float(v) for v in ep]), dilation_MALE=ratios),
          open(ROOT + "/results/analysis.json", "w"), indent=1)
print(json.dumps(ratios, indent=1))
print("decomp", [round(v, 3) for v in aD], [round(v, 3) for v in ar], [round(v, 4) for v in ep])
