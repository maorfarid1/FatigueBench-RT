"""Real-time evolution of the predictive failure-time distribution (style of Farid 2022, Fig. 7).
Top: S3 benchmark component (an S3 test component with median error); bottom: SAE keyhole experiment SM3-2
(PI-BNN fine-tuned with crack inspections, seed 0). All curves come from stored predictions; nothing is retrained."""
import json, pickle, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "realdata"))
from regen_signal import windows
from crack_obs import crack_records

plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False})
R = str(ROOT / "results") + "/"; D = str(ROOT / "data") + "/"
FR = [0.25, 0.5, 0.8, 0.95]
RED, LOAD, CRK = "#d1495b", "#e9a15b", "#1d3557"

def fit(samples):
    l = np.log(np.asarray(samples, float)[np.asarray(samples) > 0]); q05, q50, q95 = np.quantile(l, [0.05, 0.5, 0.95])
    return q50, (q95 - q05) / (2 * 1.6449)

def pdf_T(samples, t, grid):
    mu, sg = fit(samples); f = np.zeros_like(grid); m = grid > t
    f[m] = np.exp(-0.5 * ((np.log(grid[m] - t) - mu) / sg) ** 2) / ((grid[m] - t) * sg * np.sqrt(2 * np.pi))
    return f                                                  # density of T = t + RUL; zero before t (Heaviside)

def cdf_T(samples, t, grid):
    l = np.sort(np.log(np.asarray(samples, float)[np.asarray(samples) > 0])); p = (np.arange(l.size) + 0.5) / l.size
    F = np.zeros_like(grid); m = grid > t
    F[m] = np.interp(np.log(grid[m] - t), l, p, left=0.0, right=1.0)   # empirical CDF of the stored samples
    return F

def interval(samples, t):
    lo, med, hi = np.quantile(samples, [0.05, 0.5, 0.95]); return t + lo, t + med, t + hi

fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.9), sharey="row")
# ---- top row: S3 test component 2 (error at the median of the test set)
P = pickle.load(open(R + "pred_S3_iid_n600_s0_base.pkl", "rb")); c = P["c"]; idx = np.asarray(P["idx"])
S = np.asarray(P["preds"]["PI-BNN (proposed) + CP"][1]); J = 2
traj = pickle.load(open(D + "S3_test.pkl", "rb"))[J]; W = windows(traj)
m = np.where(idx == J)[0]; Kf = c["Kf"][m][0]; xmax = 1.35 * Kf
grid = np.linspace(0, xmax, 3000)
rows = [m[np.argmin(np.abs(c["tt"][m] - fr * Kf))] for fr in FR]
FMAX = max(pdf_T(S[r], c["tt"][r], grid).max() for r in rows)   # one density scale for all four panels
for a, fr, r in zip(axs[0], FR, rows):
    t = c["tt"][r]
    for k in range(int(np.ceil(t))):
        w = W[k][:512:2]; a.plot(k + np.linspace(0, 1, w.size), w, color=LOAD, lw=0.25)
    a2 = a.twinx(); f = pdf_T(S[r], t, grid); lo, med, hi = interval(S[r], t)
    a2.plot(grid, f, color=RED, lw=1.0); sel = (grid >= lo) & (grid <= hi)
    a2.fill_between(grid[sel], f[sel], color=RED, alpha=0.25, lw=0)
    a2.axvline(Kf, color="k", ls="--", lw=0.7); a2.axvline(t, color="0.45", lw=0.6)
    a2.set_ylim(0, 1.08 * FMAX); a2.set_yticks([]); a2.set_xlim(0, xmax); a2.spines["top"].set_visible(False); a.spines["top"].set_visible(False)
    a.set_title(f"$t$ = {fr:.0%} of life", fontsize=7.5, pad=5)
    a.set_xlabel("operating time (blocks)")
axs[0][0].set_ylabel("measured stress (MPa)")
axs[0][0].text(-0.36, 1.25, "", transform=axs[0][0].transAxes)
axs[0][0].text(0.97, 0.80, f"S3 component\n$K_f$ = {Kf:.0f}", transform=axs[0][0].transAxes, fontsize=6, ha="right")
# ---- bottom row: keyhole SM3-2 with crack inspections
JJ = json.load(open(R + "FT_crack_S4_k0_s0.json")); cc = {k: np.array(v) for k, v in JJ["c"].items()}; ii = np.array(JJ["idx"])
SS = pickle.load(open(R + "ftpred_crack_S4_k0_s0_PI-BNN_fine-tuned.pkl", "rb"))["scp"]
te = pickle.load(open(D + "REAL_keyhole.pkl", "rb")); ids = [t["id"] for t in te]; JK = ids.index("SM3-2")
mk = np.where(ii == JK)[0]; Kk = cc["Kf"][mk][0]; xk = 1.35 * Kk; gk = np.linspace(0, xk, 1500)
nrec, Lrec = crack_records()["SM3-2"]
for a, fr in zip(axs[1], FR):
    r = mk[np.argmin(np.abs(cc["tt"][mk] - fr * Kk))]; t = cc["tt"][r]
    seen = nrec <= t
    a.plot(nrec[seen], Lrec[seen], "o", ms=2.2, color=CRK)
    a2 = a.twinx(); F = cdf_T(SS[r], t, gk); lo, med, hi = interval(SS[r], t)
    sel = (gk >= lo) & (gk <= hi); a2.fill_between(gk[sel], 0, F[sel], color=RED, alpha=0.18, lw=0)
    a2.plot(gk, F, color=RED, lw=1.2)
    pt = float(np.mean(np.asarray(SS[r]) < Kk - t))
    a2.axvline(Kk, color="k", ls="--", lw=0.7); a2.axvline(t, color="0.45", lw=0.6)
    a2.plot([Kk], [cdf_T(SS[r], t, np.array([Kk]))[0]], "o", ms=3.2, mfc="white", mec="k", mew=0.8)
    a2.set_ylim(0, 1.05); a2.set_yticks([0, 0.5, 1]); a2.set_xlim(0, xk); a2.spines["top"].set_visible(False); a.spines["top"].set_visible(False)
    a2.tick_params(axis="y", labelsize=6.5, colors=RED)
    a.set_xlabel("operating time (passes)")
    if fr == FR[-1]: a2.set_ylabel("P(failure by $T$)", color=RED, fontsize=7)
axs[1][0].set_ylabel("crack length (mm)")
axs[1][0].text(0.04, 0.96, f"keyhole SM3-2\n$K_f$ = {Kk:,.0f}", transform=axs[1][0].transAxes, fontsize=6, ha="left", va="top")
h = [plt.Line2D([], [], color=LOAD, lw=1, label="measured stress"), plt.Line2D([], [], color=CRK, marker="o", ls="", ms=3, label="crack inspections"),
     plt.Line2D([], [], color=RED, lw=1, label="PI-BNN prediction"), plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.25, lw=0, label="90% CI (CL = 90%)"),
     plt.Line2D([], [], color="0.45", lw=0.6, label="current time $t$"), plt.Line2D([], [], color="k", ls="--", lw=0.7, label="true failure time")]
fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6.3, bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=1.6)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.4)
fig.savefig(ROOT / "tex/figs/fig_realtime_pdf.pdf"); fig.savefig(ROOT / "figs/fig_realtime_pdf.png", dpi=300)
print("Kf S3", Kf, "Kf SM3-2", Kk)
