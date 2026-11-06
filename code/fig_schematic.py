"""Fig. 1: framework schematic (matplotlib, vector PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
C = dict(data="#e8eef7", phys="#fdf0dc", nn="#e6f4ea", out="#f3e8f7", edge="#444444")


def box(x, y, w, h, text, fc, fs=6.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=C["edge"], lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, color=C["edge"]))


# measured signal
rng = np.random.default_rng(3)
t = np.linspace(0, 1, 400)
sig = np.convolve(rng.standard_normal(400), np.ones(6) / 6, "same") * (1 + 0.5 * (t > 0.5))
box(1, 21, 16, 15, "", C["data"])
ax.plot(2.5 + 12 * t, 30.5 + 2.2 * sig / np.abs(sig).max() * 1.6, lw=0.4, color="#1f4e8c")
ax.text(9, 23.5, "Measured stress window\n(one per operating block)", ha="center", fontsize=6.0)
box(1, 2, 16, 15, "Nominal material\n$\\sigma_f'$, $b$, $\\sigma_u$\n(Basquin, Goodman)", C["data"])
# features
box(20.5, 21, 18, 15, "Rainflow, Goodman,\nBasquin: damage $\\hat d_k$\nMiner state $\\hat D_k$\nmean rate $\\hat D_k/t_k$", C["phys"])
box(20.5, 2, 18, 15, "Spectral/statistical features\n(Welch, Dirlik, kurtosis,\nskewness, rate ratios)\nscale-free vector $\\mathbf{z}_k$", C["data"])
arrow(17, 30, 20.5, 30); arrow(17, 24, 20.5, 12); arrow(17, 10, 20.5, 24)
# networks
box(41.5, 21, 24, 15, "Damage-state network\n$\\log\\Delta D=\\log(1-\\hat D_k)+f_D(\\mathbf{z})$\nvariance $g_D(\\mathbf{z})$", C["nn"])
box(41.5, 2, 24, 15, "Load-forecast network\n$\\log r=\\log(\\hat D_k/t_k)+3s_k\\tanh(f_r/3)$\nvariance $g_r(\\mathbf{z})+s_k^2$", C["nn"])
arrow(38.5, 28, 41.5, 28); arrow(38.5, 10, 41.5, 10); arrow(38.5, 25, 41.5, 13); arrow(38.5, 13, 41.5, 25)
ax.text(53.5, 38.3, "Deep ensemble ($M$=5) = approximate Bayesian NN", ha="center", fontsize=7, style="italic")
# composition
box(68.5, 11, 14, 16, "Physics\ncomposition\n$\\mathrm{RUL}=\\Delta D / r$\n$\\sigma^2=\\sigma_D^2+\\sigma_r^2$", C["phys"])
arrow(65.5, 28, 68.5, 22); arrow(65.5, 10, 68.5, 16)
box(86, 11, 13, 16, "Predictive PDF\nof RUL\n+ split-conformal\ncalibration", C["out"])
arrow(82.5, 19, 86, 19)
# little pdf inset
x = np.linspace(0.01, 4, 200); y = np.exp(-(np.log(x) - 0.5) ** 2 / 0.3) / x; y = y / y.max()
ax.fill_between(86 + 3 * x, 30, 30 + 7 * y, color="#9b59b6", alpha=0.4); ax.plot(86 + 3 * x, 30 + 7 * y, color="#6c3483", lw=0.8)
ax.plot([86, 98], [30, 30], color="#444444", lw=0.5)
ax.text(92, 5.5, "updated every block\n(real time)", ha="center", fontsize=6.5)
plt.savefig(ROOT + "/figs/fig1_framework.pdf", bbox_inches="tight")
plt.savefig(ROOT + "/figs/fig1_framework.png", dpi=200, bbox_inches="tight")
