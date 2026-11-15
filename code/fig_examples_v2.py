"""Fig. 4 (v2): real-time RUL prediction for one example component per scenario.
Example selection (stated in the caption): among test components with 80 < K_f < 160 whose 90% interval covers the truth
over the last 40% of life, the one with the lowest PI-BNN log error there."""
import pickle, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
                     "font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False})
RED = "#d1495b"
BASE = {"Miner-RT (rainflow)": ("Miner-RT", "#8c8c8c"), "ANN+GPR (Farid 2022)": ("ANN+GPR [2022]", "#9467bd"),
        "Deep ensemble MLP": ("Deep ens. MLP", "#4c78a8")}
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25))
for ax, sc in zip(axs, ["S2", "S3", "S4"]):
    P = pickle.load(open(R / f"pred_{sc}_iid_n600_s0_base.pkl", "rb")); c, idx = P["c"], np.asarray(P["idx"])
    pt, smp = P["preds"]["PI-BNN (proposed) + CP"]; rul = c["rul"]; frac = c["tt"] / c["Kf"]
    lo, hi = np.quantile(smp, [0.05, 0.95], axis=1)
    ids = np.unique(idx); Kf = np.array([c["Kf"][idx == i][0] for i in ids])
    late = (frac >= 0.6) & (frac <= 0.95); e = np.abs(np.log(pt) - np.log(rul)); cov = (rul >= lo) & (rul <= hi)
    score = np.array([e[(idx == i) & late].mean() for i in ids]); cv = np.array([cov[(idx == i) & late].mean() for i in ids])
    cand = np.where((Kf > 80) & (Kf < 160) & (cv >= 0.9))[0]
    i = ids[cand[np.argmin(score[cand])]]                         # lowest late-life error with late coverage >= 90%
    print(sc, "component", i, "Kf %.0f" % c["Kf"][idx == i][0], "late MALE %.3f" % score[ids == i][0],
          "rank among %d with 80<Kf<160: %d" % (((Kf > 80) & (Kf < 160)).sum(), 1 + (score[(Kf > 80) & (Kf < 160)] < score[ids == i][0]).sum()),
          "median late MALE (all) %.3f" % np.median(score))
    m = idx == i; t = c["tt"][m]; K = c["Kf"][m][0]
    for meth, (lab, col) in BASE.items():
        ax.plot(t, P["preds"][meth][0][m], color=col, lw=0.6, alpha=0.75, label=lab)
    ax.fill_between(t, lo[m], hi[m], color=RED, alpha=0.18, lw=0, label="PI-BNN 90% CI")
    ax.plot(t, pt[m], color=RED, lw=1.3, label="PI-BNN (ours)")
    ax.plot(t, rul[m], color="k", lw=1.1, label="true RUL")
    ax.set_ylim(0, 1.6 * K); ax.set_xlim(0, K)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(f"{sc}, $K_f$ = {K:.0f} blocks", fontsize=7.5)
axs[0].set_ylabel("RUL (blocks)")
h, l = axs[0].get_legend_handles_labels(); o = [5, 4, 3, 0, 1, 2]
fig.legend([h[k] for k in o], [l[k] for k in o], loc="lower center", ncol=6, fontsize=6.3, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
fig.savefig(ROOT / "tex/figs/fig4_examples.pdf"); fig.savefig(ROOT / "figs/fig4_examples.png", dpi=300)
