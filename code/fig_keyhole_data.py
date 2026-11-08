"""Figure: the SAE keyhole real-data test set (measured histories, their rainflow spectra, crack growth)."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/paper/realdata")
import os; os.chdir("/home/claude/paper/realdata")
from parse_keyhole import history, tables
import fatigue_physics as fp

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "mathtext.fontset": "cm"})
HC = {"bracket": "#1b9e77", "suspension": "#d95f02", "transmission": "#7570b3"}
NAME = {"bracket": "GM bracket", "suspension": "Ford suspension", "transmission": "Deere transmission"}
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
ax = axs[0]
for i, h in enumerate(["bracket", "suspension", "transmission"]):
    x = history(h)[:400]
    ax.plot(np.arange(len(x)), x - 2.4 * i, color=HC[h], lw=0.5)
    ax.text(405, -2.4 * i, NAME[h], fontsize=6.3, va="center", color=HC[h])
ax.set_xlim(0, 560); ax.set_yticks([]); ax.set_xlabel("reversal (first 400 of each history)")
ax.set_ylabel("normalized load"); ax.set_title("(a) measured service histories", fontsize=7.5)
ax = axs[1]
for h in ["bracket", "suspension", "transmission"]:
    x = history(h)
    a, m, n = fp.rainflow_cycles(np.r_[x, x])
    a = 2 * a; o = np.argsort(a)[::-1]; cum = np.cumsum(n[o]) / 2
    ax.semilogx(cum, a[o], color=HC[h], lw=1, label=NAME[h])
ax.set_xlabel("cumulative cycles per pass"); ax.set_ylabel("rainflow range / peak load")
ax.legend(fontsize=5.8, loc="lower left"); ax.set_title("(b) rainflow range spectra", fontsize=7.5)
ax = axs[2]
va = {r["id"]: r for r in json.load(open("keyhole_tests.json"))["va"]}
for t in tables():
    parts = t["name"].split()
    if len(parts) < 2 or not parts[0].endswith("E"): continue
    sid = parts[1].upper().replace("-FS", "")
    if sid == "SR2-2" and np.nanmax(t["n"]) > 10000: sid = "SR3-2"
    r = va.get(sid)
    if r is None or r["censored"]: continue
    n, L = np.asarray(t["n"], float), np.asarray(t["L"], float)
    ok = np.isfinite(n) & np.isfinite(L) & (L >= 0)
    if ok.sum() < 2: continue
    ax.plot(n[ok] / r["K_f"], L[ok], color=HC[r["history"]], lw=0.6, alpha=0.7)
ax.axhline(2.54, color="0.4", ls=":", lw=0.6); ax.text(0.02, 3.5, "initiation, 2.54 mm", fontsize=5.8, color="0.3")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 55); ax.set_xlabel("life fraction $n/K_f$"); ax.set_ylabel("crack length (mm)")
ax.set_title("(c) crack growth to fracture", fontsize=7.5)
fig.tight_layout(w_pad=0.6)
fig.savefig("/home/claude/paper/tex/figs/fig_keyhole_data.pdf"); fig.savefig("/home/claude/paper/figs/fig_keyhole_data.png", dpi=300)
print("ok")
