import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False})
d = json.load(open("/home/claude/paper/results/analysis.json"))["decomp"]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(d["frac"], d["aleaD"], "o-", ms=2.5, color="#e9a03b", label="aleatoric, damage state $\\sigma_D^2$")
ax.plot(d["frac"], d["alear"], "s-", ms=2.5, color="#2a9d8f", label="aleatoric, load forecast $\\sigma_r^2$")
ax.plot(d["frac"], d["epi"], "^-", ms=2.5, color="#6c3483", label="epistemic (ensemble)")
ax.set_yscale("log"); ax.set_xlabel("life fraction $t/K_f$"); ax.set_ylabel("variance of $\\log$ RUL (median)")
ax.legend(fontsize=6.2, loc="center right")
plt.tight_layout(); plt.savefig("/home/claude/paper/figs/fig9_uq.pdf"); plt.savefig("/home/claude/paper/figs/fig9_uq.png", dpi=200)
