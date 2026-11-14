"""Real-time RUL predictions on three held-out keyhole tests: zero-shot vs adapted PI-BNN, adapted MLP."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
for _d in ("tex/figs", "figs", "results", "data"): _os.makedirs(_os.path.join(ROOT, _d), exist_ok=True)
import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "mathtext.fontset": "cm"})
R = ROOT + "/results/"
PRE, SEED = "S4", 0
J = json.load(open(R + f"FT_{PRE}_k0_s{SEED}.json"))
idx = np.array(J["idx"]); rul = np.array(J["c"]["rul"]); Kf = np.array(J["c"]["Kf"]); tt = np.array(J["c"]["tt"])
ids = pickle.load(open(R + "pred_REAL_S4_s0.pkl", "rb"))["ids"]
P = pickle.load(open(R + f"ftpred_{PRE}_k0_s{SEED}_PI-BNN_fine-tuned.pkl", "rb"))
M = pickle.load(open(R + f"ftpred_{PRE}_k0_s{SEED}_DeepensMLP_fine-tuned.pkl", "rb"))
Z = pickle.load(open(R + "pred_REAL_S4_s0.pkl", "rb"))["preds"]["PI-BNN (proposed)"]
show = ["SM3-2", "BM3-1", "TR2-2", "SR1-3"]
fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.25))
for ax, sid in zip(axs, show):
    j = ids.index(sid); m = idx == j
    f = tt[m] / Kf[m]; o = np.argsort(f); f = f[o]
    ax.plot(f, rul[m][o], color="k", lw=1.1, label="true RUL")
    lo, hi = np.quantile(P["scp"][m][o], [0.05, 0.95], 1)
    ax.fill_between(f, lo, hi, color="#d62728", alpha=0.18, lw=0, label="PI-BNN adapted, 90% PI")
    ax.plot(f, P["pt"][m][o], color="#d62728", lw=1, label="PI-BNN adapted")
    ax.plot(f, M["pt"][m][o], color="#1f77b4", lw=0.9, ls="--", label="MLP adapted")
    ax.plot(f, Z[0][m][o], color="0.5", lw=0.9, ls=":", label="PI-BNN zero-shot")
    ax.set_yscale("log"); ax.set_xlabel("life fraction"); ax.set_title(f"{sid}, $K_f$ = {Kf[m][0]:,.0f}", fontsize=7.2)
    ax.set_xlim(0, 1)
axs[0].set_ylabel("RUL (blocks)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 0.9))
fig.savefig(ROOT + "/tex/figs/fig_ft_examples.pdf"); fig.savefig(ROOT + "/figs/fig_ft_examples.png", dpi=300)
print("ok")
