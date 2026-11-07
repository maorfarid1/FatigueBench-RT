"""Figure, table and LaTeX macros for the SAE keyhole real-data test."""
import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fig_results import COL, LAB, SHORT
from real_analysis import cross_conformal, groups

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False})
R = "/home/claude/paper/results/"; RD = "/home/claude/paper/realdata/"; OUT = "/home/claude/paper/tex/"
S = json.load(open(R + "real_summary.json"))
mv = json.load(open(RD + "miner_va.json")); ls = json.load(open(RD + "local_strain_va.json"))
va = {r["id"]: r for r in json.load(open(RD + "keyhole_tests.json"))["va"]}
BASE = ["Miner-RT (rainflow)", "Dirlik-RT (spectral)", "ANN+GPR (Farid 2022)", "Deep ensemble MLP",
        "Deep ensemble LSTM", "PI-BNN (proposed)"]
HMARK = {"bracket": "s", "suspension": "o", "transmission": "^"}

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1.05, 1.15, 1.0]))
# (a) total life: CA-calibrated Miner and FEA-based local strain, no VA data used
ax = axs[0]
lim = [0.5, 3e5]
ax.plot(lim, lim, color="0.3", lw=0.6)
for f in (2, 3):
    ax.plot(lim, [l * f for l in lim], color="0.6", lw=0.5, ls=":" if f == 3 else "--")
    ax.plot(lim, [l / f for l in lim], color="0.6", lw=0.5, ls=":" if f == 3 else "--")
for q, q2 in zip(mv, ls["va"]):
    if q["censored"]: continue
    r = va[q["id"]]; c = "#d62728" if r["mat"] == "M" else "#1f77b4"
    ax.plot(q["K_miner"], q["K_f"], HMARK[r["history"]], ms=3.2, mfc=c, mec="none", alpha=0.85)
    ax.plot(q2["K_ls"], q2["K_f"], HMARK[r["history"]], ms=3.2, mfc="none", mec=c, mew=0.6, alpha=0.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted life (blocks)"); ax.set_ylabel("observed life (blocks)")
h = [plt.Line2D([], [], marker=HMARK[k], ls="", color="0.3", ms=3.2, label=k) for k in HMARK]
h += [plt.Line2D([], [], marker="o", ls="", mfc="#d62728", mec="none", ms=3.2, label="Man-Ten"),
      plt.Line2D([], [], marker="o", ls="", mfc="#1f77b4", mec="none", ms=3.2, label="RQC-100"),
      plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", ms=3.2, label="FE + Neuber + SWT")]
ax.legend(handles=h, fontsize=5.6, loc="upper left", handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
ax.set_title("(a) total life, 47 VA tests", fontsize=7.5)

# (b) interval coverage: simulation-calibrated vs real-data (leave-one-group-out) conformal
ax = axs[1]
x = np.arange(len(BASE)); w = 0.38
raw = [S["methods"][m]["PICP90"][0] for m in BASE]
cp = [S["methods"][m + " + CP(real)"]["PICP90"][0] for m in BASE]
ax.bar(x - w / 2, raw, w, color=[COL[m] for m in BASE], alpha=0.45, lw=0)
ax.bar(x + w / 2, cp, w, color=[COL[m] for m in BASE], lw=0)
ax.axhline(0.9, color="0.4", ls="--", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([LAB[m].replace(" [2022]", "").replace("Deep ens. ", "") for m in BASE], rotation=35, ha="right", fontsize=6.3)
ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("PICP90 on real tests")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.45, lw=0, label="simulation-calibrated"),
                   plt.Rectangle((0, 0), 1, 1, color="0.5", lw=0, label="+ conformal on other real groups")],
          fontsize=5.6, loc="upper left", borderaxespad=0.2, ncol=1)
ax.set_title("(b) 90% interval coverage", fontsize=7.5)

# (c) median bias of log RUL by test life: black boxes collapse on long lives
ax = axs[2]
bins = [(0, 30), (30, 1000), (1000, 1e9)]
for m in ["Miner-RT (rainflow)", "Deep ensemble MLP", "Deep ensemble LSTM", "ANN+GPR (Farid 2022)", "PI-BNN (proposed)"]:
    ys = []
    for sd in (0, 1, 2):
        d = pickle.load(open(R + f"pred_REAL_S4_s{sd}.pkl", "rb")); c = d["c"]
        frac = c["tt"] / c["Kf"]; sel = (frac >= 0.1) & (frac <= 0.95)
        e = np.log(d["preds"][m][0].astype(float)) - np.log(c["rul"])
        ys.append([np.mean(e[sel & (c["Kf"] >= a) & (c["Kf"] < b)]) for a, b in bins])
    y = np.mean(ys, 0)
    ax.plot(range(3), y, "o-", ms=2.8, lw=1, color=COL[m], label=LAB[m])
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["<30", "30 to 1000", ">1000"])
ax.set_xlabel("test life (blocks)"); ax.set_ylabel("mean error of log RUL")
ax.legend(fontsize=5.6, loc="lower left", borderaxespad=0.2, labelspacing=0.25)
ax.set_title("(c) bias vs life scale", fontsize=7.5)
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT + "figs/fig10_real.pdf"); fig.savefig("/home/claude/paper/figs/fig10_real.png", dpi=400)

# bias table numbers for the text
bias = {}
for m in BASE:
    ys = []
    for sd in (0, 1, 2):
        d = pickle.load(open(R + f"pred_REAL_S4_s{sd}.pkl", "rb")); c = d["c"]
        frac = c["tt"] / c["Kf"]; sel = (frac >= 0.1) & (frac <= 0.95)
        e = np.log(d["preds"][m][0].astype(float)) - np.log(c["rul"])
        l, h = np.quantile(d["preds"][m][1].astype(float), [0.05, 0.95], 1)
        cov = (c["rul"] >= l) & (c["rul"] <= h)
        k = sel & (c["Kf"] >= 1000)
        ys.append([np.mean(e[k]), np.mean(np.abs(e[k])), np.mean(cov[k])])
    bias[m] = np.mean(ys, 0).tolist()
json.dump(bias, open(R + "real_longlife.json", "w"), indent=1)

# table
rows = []
for m in BASE:
    for suf, lab in [("", SHORT[m]), (" + CP(real)", "\\quad + CP (real)")]:
        r = S["methods"][m + suf]
        rows.append(f"{lab} & {r['CRPS'][0]:.2f} & {r['MALE'][0]:.2f} & {r['PICP90'][0]:.2f} & {r['MPIW90'][0]:.1f} & {r['ECE'][0]:.3f} \\\\")
tab = r"""\begin{table}[t]
\centering
\caption{Simulation-to-real test on the 47 SAE keyhole variable-amplitude tests that failed (three service histories, two steels, lives from 2.2 to 57,090 blocks). All learned models are trained on FatigueBench-RT S4 only; mean of 3 seeds. ``+ CP (real)'': split-conformal rescaling calibrated, for each history and steel group, on the five other groups (no data of the tested group). CRPS and MPIW90 on log RUL.}
\label{tab:real}
\small
\begin{tabular}{lccccc}
\toprule
Method & CRPS & MALE & PICP90 & MPIW90 & ECE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(OUT + "tab_real.tex", "w").write(tab)

L = S["life"]
mac = {
    "realNfail": L["n_fail"], "realNca": L["n_ca"], "realSd": f"{L['sd_log10']:.2f}", "realBias": f"{L['bias_log10']:+.2f}",
    "realWithinThree": f"{100 * L['within3']:.0f}", "realWithinTwo": f"{100 * L['within2']:.0f}",
    "realRepSd": f"{L['rep_sd_log10']:.2f}", "realLifeMin": f"{L['life_min']:.1f}", "realLifeMax": f"{L['life_max']:,.0f}",
    "realCaRmsM": f"{L['calib']['M']['rms_log10']:.2f}", "realCaRmsR": f"{L['calib']['R']['rms_log10']:.2f}",
    "realbM": f"{L['calib']['M']['b']:.2f}", "realbR": f"{L['calib']['R']['b']:.2f}",
}
rl = np.array(ls["ca_log10_ratio"])
lsr = np.log10([q["K_f"] / q["K_ls"] for q in ls["va"] if not q["censored"]])
mac.update(lsCaM=f"{np.mean(np.abs(rl[:6]) < np.log10(2)) * 6:.0f}", lsCaRbias=f"{rl[6:].mean():.2f}",
           lsVaBias=f"{lsr.mean():+.2f}", lsVaSd=f"{lsr.std(ddof=1):.2f}", lsVaWithinThree=f"{100 * np.mean(np.abs(lsr) < np.log10(3)):.0f}")
for m, key in [("Deep ensemble MLP", "Mlp"), ("Deep ensemble LSTM", "Lstm"), ("PI-BNN (proposed)", "Pibnn"),
               ("Miner-RT (rainflow)", "Miner"), ("ANN+GPR (Farid 2022)", "Farid")]:
    b = bias[m]
    mac[f"long{key}Bias"] = f"{b[0]:+.1f}"; mac[f"long{key}Male"] = f"{b[1]:.2f}"; mac[f"long{key}Cov"] = f"{b[2]:.2f}"
    r = S["methods"][m]
    mac[f"real{key}Crps"] = f"{r['CRPS'][0]:.2f}"; mac[f"real{key}Cov"] = f"{r['PICP90'][0]:.2f}"
    rc = S["methods"][m + " + CP(real)"]
    mac[f"real{key}CpCov"] = f"{rc['PICP90'][0]:.2f}"; mac[f"real{key}CpCrps"] = f"{rc['CRPS'][0]:.2f}"; mac[f"real{key}Lam"] = f"{rc['lambda']:.1f}"
with open(OUT + "numbers_real.tex", "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(mac, indent=0))
