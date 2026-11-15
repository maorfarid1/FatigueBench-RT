"""Real-time evolution of the predictive failure-time distribution (style of Farid 2022, Fig. 7).
Top: S3 benchmark component (the median-error component used in Fig. 8); bottom: SAE keyhole experiment SM3-2
(PI-BNN fine-tuned with crack inspections, seed 0). All curves come from stored predictions; nothing is retrained."""
import json, pickle, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
sys.path.insert(0, "/home/claude/paper/realdata")
from regen_signal import windows
from crack_obs import crack_records

plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False})
R = "/home/claude/paper/results/"; D = "/home/claude/paper/data/"
FR = [0.25, 0.5, 0.8, 0.95]
RED, LOAD, CRK = "#d1495b", "#e9a15b", "#1d3557"

def pdf_T(samples, t, grid):
    r = np.asarray(samples, float); r = r[r > 0]
    k = gaussian_kde(np.log(r), bw_method=0.35)
    f = np.zeros_like(grid); m = grid > t
    f[m] = k(np.log(grid[m] - t)) / (grid[m] - t)          # density of T = t + RUL; zero before t (Heaviside)
    return f

def interval(samples, t):
    lo, med, hi = np.quantile(samples, [0.05, 0.5, 0.95]); return t + lo, t + med, t + hi

fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.9), sharey="row")
# ---- top row: S3 component 17
P = pickle.load(open(R + "pred_S3_iid_n600_s0_base.pkl", "rb")); c = P["c"]; idx = np.asarray(P["idx"])
S = np.asarray(P["preds"]["PI-BNN (proposed) + CP"][1]); J = 17
traj = pickle.load(open(D + "S3_test.pkl", "rb"))[J]; W = windows(traj)
m = np.where(idx == J)[0]; Kf = c["Kf"][m][0]; xmax = 1.35 * Kf
grid = np.linspace(0, xmax, 1500)
for a, fr in zip(axs[0], FR):
    r = m[np.argmin(np.abs(c["tt"][m] - fr * Kf))]; t = c["tt"][r]
    for k in range(int(np.ceil(t))):
        w = W[k][:512:2]; a.plot(k + np.linspace(0, 1, w.size), w, color=LOAD, lw=0.25)
    a2 = a.twinx(); f = pdf_T(S[r], t, grid); lo, med, hi = interval(S[r], t)
    a2.plot(grid, f, color=RED, lw=1.0); sel = (grid >= lo) & (grid <= hi)
    a2.fill_between(grid[sel], f[sel], color=RED, alpha=0.25, lw=0)
    a2.axvline(Kf, color="k", ls="--", lw=0.7); a2.axvline(t, color="0.45", lw=0.6)
    a2.set_ylim(0, 1.15 * f.max()); a2.set_yticks([]); a2.set_xlim(0, xmax); a2.spines["top"].set_visible(False); a.spines["top"].set_visible(False)
    a.set_title(f"$t$ = {fr:.0%} of life", fontsize=7.5, pad=11)
    a2.text(0.5, 1.02, f"predicted {med:,.0f}, 90% CI [{lo:,.0f}, {hi:,.0f}]", transform=a2.transAxes, ha="center", va="bottom", fontsize=5.6, color=RED)
    a.set_xlabel("operating time (blocks)")
axs[0][0].set_ylabel("measured stress (MPa)")
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
    a2 = a.twinx(); f = pdf_T(SS[r], t, gk); lo, med, hi = interval(SS[r], t)
    a2.plot(gk, f, color=RED, lw=1.0); sel = (gk >= lo) & (gk <= hi)
    a2.fill_between(gk[sel], f[sel], color=RED, alpha=0.25, lw=0)
    a2.axvline(Kk, color="k", ls="--", lw=0.7); a2.axvline(t, color="0.45", lw=0.6)
    a2.set_ylim(0, 1.15 * f.max()); a2.set_yticks([]); a2.set_xlim(0, xk); a2.spines["top"].set_visible(False); a.spines["top"].set_visible(False)
    a2.text(0.5, 1.02, f"predicted {med/1e3:.1f}k, 90% CI [{lo/1e3:.1f}k, {hi/1e3:.1f}k]", transform=a2.transAxes, ha="center", va="bottom", fontsize=5.6, color=RED)
    a.set_xlabel("operating time (passes)")
axs[1][0].set_ylabel("crack length (mm)")
axs[1][0].text(0.97, 0.80, f"keyhole SM3-2\n$K_f$ = {Kk:,.0f}", transform=axs[1][0].transAxes, fontsize=6, ha="right")
h = [plt.Line2D([], [], color=LOAD, lw=1, label="measured stress"), plt.Line2D([], [], color=CRK, marker="o", ls="", ms=3, label="crack inspections"),
     plt.Line2D([], [], color=RED, lw=1, label="PI-BNN failure-time PDF"), plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.25, lw=0, label="90% CI (CL = 90%)"),
     plt.Line2D([], [], color="0.45", lw=0.6, label="current time $t$"), plt.Line2D([], [], color="k", ls="--", lw=0.7, label="true failure time")]
fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.4)
fig.savefig("/home/claude/paper/tex/figs/fig_realtime_pdf.pdf"); fig.savefig("/home/claude/paper/figs/fig_realtime_pdf.png", dpi=300)
print("Kf S3", Kf, "Kf SM3-2", Kk)
