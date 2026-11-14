"""Figs. 2-3: benchmark illustration and statistics."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gen, fatigue_physics as fp
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})
D = ROOT + "/data/"
SC = ["S1", "S2", "S3", "S4"]
TITLE = {"S1": "S1 stationary Gaussian", "S2": "S2 mission mix", "S3": "S3 + non-Gaussian", "S4": "S4 + drift, sequence"}
COL = {"S1": "#1b6ca8", "S2": "#2a9d8f", "S3": "#e9a03b", "S4": "#c8553d"}

fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6))
for j, sc in enumerate(SC):
    cfg = gen.SCEN[sc]
    rng = np.random.default_rng(7 + j)
    regs = [gen.new_regime(cfg, rng, (3.0, 7.0), 800.0) for _ in range(3 if cfg["nonstat"] else 1)]
    seq = [0, 0, 1, 2, 2, 1] if cfg["nonstat"] else [0] * 6
    t0 = 0.0
    for b, ri in enumerate(seq):
        r = regs[ri]
        x = r["mfrac"] * 800 + 120 * r["rho"] * gen.unit_window(r, rng)
        n = 512
        tt = t0 + np.arange(n) / fp.FS
        axs[0, j].plot(tt, x[:n], lw=0.3, color=COL[sc])
        axs[0, j].axvline(t0, color="0.6", lw=0.4, ls=":")
        t0 = tt[-1] + 1 / fp.FS
    axs[0, j].set_title(TITLE[sc], fontsize=7.5)
    axs[0, j].set_xlabel("time (s), 1 s from each of 6 blocks", fontsize=6.5)
    if j == 0: axs[0, j].set_ylabel("stress (MPa)")
    # example trajectory
    T = pickle.load(open(D + f"{sc}_test.pkl", "rb"))
    lives = np.array([t["K_f"] for t in T])
    t = T[int(np.argsort(np.abs(lives - 120))[0])]
    k = np.arange(1, len(t["d_rf"]) + 1)
    Dn = np.cumsum(t["d_rf"])
    if sc == "S4":
        # true damage under DCA is not stored; show true Miner sum (linear) for reference
        Dt = np.cumsum(t["d_true"])
        lab = "true linear damage sum"
    else:
        Dt = np.cumsum(t["d_true"]); lab = "true damage"
    ax = axs[1, j]
    ax.plot(k, Dn, color=COL[sc], lw=1.0, label="nominal Miner $\\hat D_k$")
    ax.plot(k, Dt, color="k", lw=0.8, ls="--", label=lab)
    ax.axvline(t["K_f"], color="r", lw=0.8)
    ax.text(t["K_f"], 0.05, " failure", color="r", fontsize=6.5, rotation=90, va="bottom", ha="right")
    ax2 = ax.twinx()
    ax2.bar(k, t["d_rf"], color="0.75", width=1.0, zorder=0)
    ax2.set_yscale("log"); ax2.tick_params(labelsize=6)
    if j == 3: ax2.set_ylabel("block damage $\\hat d_k$", fontsize=7)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlabel("block $k$ (h)")
    if j == 0:
        ax.set_ylabel("cumulative damage"); ax.legend(loc="upper left", fontsize=6)
plt.tight_layout()
plt.savefig(ROOT + "/figs/fig2_scenarios.pdf"); plt.savefig(ROOT + "/figs/fig2_scenarios.png", dpi=200)

# ---- Fig 3 statistics
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
stats = {}
for sc in SC:
    T = pickle.load(open(D + f"{sc}_train.pkl", "rb")) + pickle.load(open(D + f"{sc}_val.pkl", "rb")) + pickle.load(open(D + f"{sc}_test.pkl", "rb"))
    life = np.array([t["K_f"] for t in T])
    ms = []
    for t in T:
        kf = int(np.floor(t["K_f"]))
        ms.append(t["d_rf"][:kf].sum() + t["d_rf"][min(kf, len(t["d_rf"]) - 1)] * (t["K_f"] - kf))
    ratio = np.array([np.log10(t["d_dk"].sum() / t["d_rf"].sum()) for t in T])
    stats[sc] = dict(life=life, ms=np.array(ms), ratio=ratio, n=len(T), nblk=int(sum(len(t["d_rf"]) for t in T)))
bins = np.logspace(np.log10(15), np.log10(1500), 30)
for sc in SC:
    axs[0].hist(stats[sc]["life"], bins=bins, histtype="step", color=COL[sc], lw=1.0, label=sc)
axs[0].set_xscale("log"); axs[0].set_xlabel("life to failure $K_f$ (blocks)"); axs[0].set_ylabel("components"); axs[0].legend(fontsize=6)
axs[1].boxplot([stats[sc]["ms"] for sc in SC], tick_labels=SC, showfliers=False, widths=0.5)
axs[1].axhline(1, color="0.5", lw=0.6, ls="--"); axs[1].set_yscale("log"); axs[1].set_ylabel("nominal Miner sum at failure")
axs[2].boxplot([stats[sc]["ratio"] for sc in SC], tick_labels=SC, showfliers=False, widths=0.5)
axs[2].axhline(0, color="0.5", lw=0.6, ls="--"); axs[2].set_ylabel("$\\log_{10}$(Dirlik / rainflow damage)")
plt.tight_layout()
plt.savefig(ROOT + "/figs/fig3_stats.pdf"); plt.savefig(ROOT + "/figs/fig3_stats.png", dpi=200)
import json
json.dump({sc: dict(n=stats[sc]["n"], nblk=stats[sc]["nblk"], life_med=float(np.median(stats[sc]["life"])),
                    life_p5=float(np.percentile(stats[sc]["life"], 5)), life_p95=float(np.percentile(stats[sc]["life"], 95)),
                    ms_med=float(np.median(stats[sc]["ms"])), ms_p5=float(np.percentile(stats[sc]["ms"], 5)),
                    ms_p95=float(np.percentile(stats[sc]["ms"], 95)), ratio_med=float(10 ** np.median(stats[sc]["ratio"])))
           for sc in SC}, open(ROOT + "/results/data_stats.json", "w"), indent=1)
print(open(ROOT + "/results/data_stats.json").read())
