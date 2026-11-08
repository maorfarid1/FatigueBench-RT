"""Illustrations of the benchmark ingredients (Hermite model, spectra, S-N/Goodman, damage curve approach,
drift) and of the PI-BNN rate correction."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fatigue_physics as fp
from scipy.stats import norm

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "mathtext.fontset": "cm"})
OUT = "/home/claude/paper/tex/figs/"; PNG = "/home/claude/paper/figs/"
C = plt.cm.viridis(np.linspace(0.05, 0.85, 4))

fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.3))
# (a) Hermite map
u = np.linspace(-4, 4, 400)
ax = axs[0, 0]
for c, g4 in zip(C, [3, 5, 7, 10]):
    h3, h4, k = fp.hermite_coeffs(0.0, g4)
    ax.plot(u, k * (u + h3 * (u ** 2 - 1) + h4 * (u ** 3 - 3 * u)), color=c, lw=1, label=rf"$\gamma_4={g4}$")
h3, h4, k = fp.hermite_coeffs(0.4, 7)
ax.plot(u, k * (u + h3 * (u ** 2 - 1) + h4 * (u ** 3 - 3 * u)), color="#d62728", lw=1, ls="--", label=r"$\gamma_4=7,\ \gamma_3=0.4$")
ax.set_xlabel(r"Gaussian input $u$"); ax.set_ylabel(r"$h(u)$"); ax.legend(fontsize=6, loc="upper left")
ax.set_title("(a) Hermite transformation", fontsize=7.5)
# (b) PDFs
ax = axs[0, 1]
rs = np.random.default_rng(0); z = rs.standard_normal(4_000_000)
bins = np.linspace(-8, 8, 161); xc = 0.5 * (bins[1:] + bins[:-1])
for c, g4 in zip(C, [3, 5, 7, 10]):
    y = fp.hermite_transform(z, 0.0, g4)
    hst, _ = np.histogram(y, bins, density=True); m = hst > 0
    ax.semilogy(xc[m], hst[m], color=c, lw=1, label=rf"$\gamma_4={g4}$")
ax.set_ylim(1e-6, 1); ax.set_xlim(-8, 8); ax.set_xlabel("standardized stress"); ax.set_ylabel("probability density")
ax.set_title("(b) heavier tails with kurtosis", fontsize=7.5); ax.legend(fontsize=6, loc="lower center")
# (c) PSD and a sample window
ax = axs[0, 2]
f = np.linspace(0, 120, 600)
for c, pk in zip(C[:3], [[(12, 3, 1.0)], [(35, 8, 0.6), (70, 10, 0.4)], [(60, 15, 1.0)]]):
    g = fp.psd_shape(f, pk); ax.plot(f, g / g.max(), color=c, lw=1)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("normalized PSD"); ax.set_ylim(0, 1.15)
ax.set_title("(c) regime spectra (examples)", fontsize=7.5)
ins = ax.inset_axes([0.56, 0.62, 0.42, 0.33])
x = fp.synth_gaussian([(35, 8, 0.6), (70, 10, 0.4)], np.random.default_rng(3))[:512]
ins.plot(np.arange(512) / fp.FS, fp.hermite_transform(x, 0.2, 7), lw=0.4, color="0.2")
ins.set_xticks([]); ins.set_yticks([]); 
# (d) S-N and Goodman
ax = axs[1, 0]
N = np.logspace(3, 8, 200)
for (sf, b, lab, c) in [(900, -0.07, r"$\sigma_f'=900,\ b=-0.07$", C[0]), (1600, -0.15, r"$\sigma_f'=1600,\ b=-0.15$", C[2]),
                        (917, -0.095, "Man-Ten", "#d62728"), (1160, -0.075, "RQC-100", "#1f77b4")]:
    ax.loglog(N, sf * (2 * N) ** b, color=c, lw=1, ls="-" if "sigma" in lab else "--", label=lab)
ax.set_xlabel("cycles to failure $N$"); ax.set_ylabel(r"$\sigma_{ar}$ (MPa)"); ax.legend(fontsize=5.8, loc="lower left")
ax.set_title("(d) Basquin curves (range and SAE steels)", fontsize=7.5)
# (e) damage curve approach: two-level sequences
ax = axs[1, 1]
def two_level(q1, q2, frac1=0.3):
    r1 = np.linspace(0, frac1, 100); D1 = r1 ** q1
    rem = 1 - D1[-1] ** (1 / q2)            # equivalent fraction left at level 2
    r2 = np.linspace(0, rem, 100); D2 = ((D1[-1]) ** (1 / q2) + r2) ** q2
    return np.r_[r1, frac1 + r2], np.r_[D1, D2], frac1 + rem
x, D, L = two_level(1, 1); ax.plot(x, D, color="0.4", lw=1, label=f"Miner, life {L:.2f}")
x, D, L = two_level(2.0, 0.5); ax.plot(x, D, color="#d62728", lw=1, label=f"low then high, life {L:.2f}")
x, D, L = two_level(0.5, 2.0); ax.plot(x, D, color="#1f77b4", lw=1, label=f"high then low, life {L:.2f}")
ax.axvline(0.3, color="0.7", lw=0.5, ls=":")
ax.set_xlabel(r"cycle ratio $\sum n_i/N_i$"); ax.set_ylabel("damage $D$"); ax.set_ylim(0, 1.02)
ax.legend(fontsize=5.8, loc="upper left"); ax.set_title("(e) damage curve approach, two levels", fontsize=7.5)
# (f) drift
ax = axs[1, 2]
k = np.linspace(0, 1.5, 200)
for fe, c in zip([0.75, 1.0, 1.35], [C[0], "0.5", C[2]]):
    ax.plot(k, 1 + (fe - 1) * (1 - np.exp(-k / 0.3)), color=c, lw=1, label=rf"$f_{{\mathrm{{end}}}}={fe}$")
ax.set_xlabel(r"block $k$ / target life $L^\ast$"); ax.set_ylabel("load factor $a_k$"); ax.legend(fontsize=6, loc="center right")
ax.set_title("(f) load drift in S4", fontsize=7.5)
fig.tight_layout(h_pad=0.9, w_pad=0.6)
fig.savefig(OUT + "fig_ingredients.pdf"); fig.savefig(PNG + "fig_ingredients.png", dpi=300)

# PI-BNN rate correction: bounded and shrinking with the standard error
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.65))
fr = np.linspace(-8, 8, 200)
for s, c in zip([1.0, 0.3, 0.1], C[:3]):
    axs[0].plot(fr, 3 * s * np.tanh(fr / 3), color=c, lw=1, label=rf"$s_k={s}$")
axs[0].set_xlabel(r"network output $f_r$"); axs[0].set_ylabel("correction of log rate"); axs[0].legend(fontsize=5.8)
axs[0].set_title(r"(a) $3s_k\tanh(f_r/3)$", fontsize=7)
t = np.logspace(0, 3, 100)
for v, c in zip([0.2, 1.0, 3.0], C[:3]):
    axs[1].loglog(t, np.sqrt((v + 1) / t), color=c, lw=1, label=rf"$v_k={v}$")
axs[1].set_xlabel("observed blocks $t_k$"); axs[1].set_ylabel("$s_k$"); axs[1].legend(fontsize=5.8)
axs[1].set_title(r"(b) $s_k=\sqrt{(v_k+1)/t_k}$", fontsize=7)
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT + "fig_shrink.pdf"); fig.savefig(PNG + "fig_shrink.png", dpi=300)
print("ok")
