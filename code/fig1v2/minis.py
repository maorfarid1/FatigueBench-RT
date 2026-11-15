"""Mini plots for Fig. 1 v3, drawn from benchmark component S3/17 and its stored PI-BNN prediction."""
import sys, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde
sys.path.insert(0, "/home/claude/paper/code"); from regen_signal import windows
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
                     "font.size": 11.5, "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
O = "/home/claude/paper/code/fig1v2/"
NAVY, SAND, CORAL, TEAL = "#1d3557", "#e9a15b", "#d1495b", "#2a9d8f"
traj = pickle.load(open("/home/claude/paper/data/S3_test.pkl", "rb"))[17]; W = windows(traj)
# 1. measured signal (one block = 1 s window segment) for blocks 0..5
fig, ax = plt.subplots(figsize=(2.1, 1.2), dpi=400)
x = np.concatenate([w[:512] for w in W[:6]]); t = np.arange(x.size) / 512
ax.plot(t, x, color=SAND, lw=0.35); ax.set_xlim(0, 6); ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([0, 200])
ax.set_xlabel("time (s)", labelpad=1); ax.set_ylabel("$x_k(t)$", labelpad=4)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2); fig.savefig(O + "m_signal.png", transparent=True); plt.close(fig)
# 2. Miner state vs block (nominal) with true failure
fig, ax = plt.subplots(figsize=(1.8, 1.15), dpi=400)
D = np.cumsum(traj["d_rf"]); k = np.arange(1, D.size + 1)
ax.plot(k, D, color=NAVY, lw=1.2); ax.fill_between(k, 0, D, color=NAVY, alpha=0.08)
ax.set_xlim(0, k[-1]); ax.set_ylim(0, None); ax.set_xticks([0, 40, 80]); ax.set_yticks([0, 0.5, 1.0] if D.max() > 1 else [0, 0.5])
ax.set_xlabel("block $k$", labelpad=1); ax.set_ylabel(r"$\hat D_k$", labelpad=1)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2); fig.savefig(O + "m_miner.png", transparent=True); plt.close(fig)
# 3. predictive density of failure time at 50% life (stored PI-BNN + CP prediction)
P = pickle.load(open("/home/claude/paper/results/pred_S3_iid_n600_s0_base.pkl", "rb")); c = P["c"]; idx = np.asarray(P["idx"])
S = np.asarray(P["preds"]["PI-BNN (proposed) + CP"][1]); m = np.where(idx == 17)[0]; Kf = c["Kf"][m][0]
r = m[np.argmin(np.abs(c["tt"][m] - 0.5 * Kf))]; t0 = c["tt"][r]; sm = S[r][S[r] > 0]
_l=np.log(sm[sm>0]); _q05,_q50,_q95=np.quantile(_l,[.05,.5,.95]); _sd=(_q95-_q05)/(2*1.6449); g = np.linspace(0, 1.6 * Kf, 800); f = np.zeros_like(g); mm = g > t0
f[mm] = np.exp(-0.5*((np.log(g[mm]-t0)-_q50)/_sd)**2) / (g[mm] - t0); lo, hi = t0 + np.quantile(sm, [0.05, 0.95])
fig, ax = plt.subplots(figsize=(2.0, 1.6), dpi=400)
ax.plot(g, f, color=CORAL, lw=1.3); sel = (g >= lo) & (g <= hi); ax.fill_between(g[sel], f[sel], color=CORAL, alpha=0.22, lw=0)
ax.axvline(t0, color="0.35", lw=0.7, ls=(0, (3, 2))); ax.set_ylim(0, 1.15 * f.max()); ax.set_xlim(0, g[-1]); ax.set_yticks([])
ax.set_xticks([0, 50, 100]); ax.set_xlabel("failure time $T$", labelpad=1); ax.text(t0, 1.17 * f.max(), "$t$", ha="center", va="bottom", fontsize=11.5)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2); fig.savefig(O + "m_pdf.png", transparent=True); plt.close(fig)
# 4. specimen from Farid (2022) Fig. 6, native resolution
Image.open(O + "src_2021_fig6.png").crop((22, 2, 158, 148)).save(O + "m_spec.png")
print("ok", t0, lo, hi, Kf)
