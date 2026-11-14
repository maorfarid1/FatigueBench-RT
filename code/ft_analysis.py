"""Aggregate the fine-tuning (few-shot transfer) runs on the SAE keyhole tests: table, figure, macros."""
import glob, json, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42})
R = "/home/claude/paper/results/"; OUT = "/home/claude/paper/tex/"
KS = [3, 5, 10, 20, 0]
METH = ["Miner-RT, recalibrated on real", "ANN+GPR, trained on real", "Deep ens. MLP, trained on real",
        "PI-BNN, trained on real", "Deep ens. MLP, fine-tuned", "PI-BNN, fine-tuned"]
COL = {"Miner-RT, recalibrated on real": "#7f7f7f", "ANN+GPR, trained on real": "#9467bd",
       "Deep ens. MLP, trained on real": "#1f77b4", "PI-BNN, trained on real": "#d62728",
       "Deep ens. MLP, fine-tuned": "#1f77b4", "PI-BNN, fine-tuned": "#d62728"}
LS = {m: ("--" if "trained on real" in m else "-") for m in METH}
LAB = {"Miner-RT, recalibrated on real": "Miner-RT, recalibrated", "ANN+GPR, trained on real": "ANN+GPR, real only",
       "Deep ens. MLP, trained on real": "MLP, real only", "PI-BNN, trained on real": "PI-BNN, real only",
       "Deep ens. MLP, fine-tuned": "MLP, sim + fine-tuned", "PI-BNN, fine-tuned": "PI-BNN, sim + fine-tuned"}


def load(pre):
    out = {}
    for f in glob.glob(R + f"FT_{pre}_k*_s*.json"):
        d = json.load(open(f)); k = int(re.search(r"_k(\d+)_", f).group(1))
        for m, r in d["results"].items():
            out.setdefault((m, k), []).append(r)
    return out


def stat(rows, key):
    v = np.array([r[key] for r in rows]); return float(v.mean()), float(v.std()), len(v)


def nspec(k):
    return k if k else 39          # about 39 to 41 specimens remain when one group is held out


if __name__ == "__main__":
    res = {pre: load(pre) for pre in ["S4", "S4_wide"]}
    for pre, d in res.items():
        print("==", pre)
        for m in METH:
            line = f"{m:34s}"
            for k in KS:
                if (m, k) in d:
                    c, cs, n = stat(d[(m, k)], "CRPS"); pc, _, _ = stat(d[(m + " + CP", k)], "PICP90")
                    line += f" | k={k or 'all'}: {c:.2f}±{cs:.2f} cov {pc:.2f} (n{n})"
            print(line)
    json.dump({pre: {f"{m}|{k}": {key: stat(v, key)[:2] for key in ["CRPS", "MALE", "PICP90", "MPIW90", "ECE"]}
                     for (m, k), v in d.items()} for pre, d in res.items()}, open(R + "ft_summary.json", "w"), indent=1)


def make_outputs(pre_main="S4", pre_alt="S4_wide"):
    res = {pre: load(pre) for pre in [pre_main, pre_alt]}
    d, w = res[pre_main], res[pre_alt]
    xs = [3, 5, 10, 20, 39]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35))
    def curve(dd, m, key, cp=False):
        mm = m + (" + CP" if cp else "")
        return np.array([stat(dd[(mm, k)], key)[0] if (mm, k) in dd else np.nan for k in KS]), \
               np.array([stat(dd[(mm, k)], key)[1] if (mm, k) in dd else np.nan for k in KS])
    for m in METH:
        mu, sd = curve(d, m, "CRPS")
        axs[0].plot(xs, mu, marker="o", ms=2.6, lw=1, ls=LS[m], color=COL[m], label=LAB[m])
        if "fine-tuned" in m: axs[0].fill_between(xs, mu - sd, mu + sd, color=COL[m], alpha=0.15, lw=0)
        axs[1].plot(xs, curve(d, m, "PICP90", True)[0], marker="o", ms=2.6, lw=1, ls=LS[m], color=COL[m])
        axs[2].plot(xs, curve(d, m, "MPIW90", True)[0], marker="o", ms=2.6, lw=1, ls=LS[m], color=COL[m])
    for m in ["Deep ens. MLP, fine-tuned", "PI-BNN, fine-tuned"]:
        axs[0].plot(xs, curve(w, m, "CRPS")[0], marker="^", ms=2.4, lw=0.7, ls=":", color=COL[m],
                    label=LAB[m].replace("sim", "wide sim"))
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["3", "5", "10", "20", "all"]); ax.minorticks_off()
        ax.set_xlabel("real tests used for adaptation")
    axs[0].set_yscale("log"); axs[0].set_ylabel("CRPS of log RUL"); axs[0].set_title("(a) accuracy", fontsize=7.5)
    axs[0].set_yticks([0.5, 1, 2, 5]); axs[0].set_yticklabels(["0.5", "1", "2", "5"]); axs[0].set_ylim(0.45, 12)
    axs[1].axhline(0.9, color="0.4", ls="--", lw=0.6); axs[1].set_ylim(0.2, 1.02); axs[1].set_ylabel("PICP90 after conformal")
    axs[1].set_title("(b) coverage", fontsize=7.5)
    axs[2].set_yscale("log"); axs[2].set_yticks([5, 10, 20, 40]); axs[2].set_yticklabels(["5", "10", "20", "40"]); axs[2].minorticks_off(); axs[2].set_ylabel("MPIW90 (log RUL)"); axs[2].set_title("(c) interval width", fontsize=7.5)
    axs[0].legend(fontsize=5.2, loc="upper right", labelspacing=0.15)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT + "figs/fig_finetune.pdf"); fig.savefig("/home/claude/paper/figs/fig_finetune.png", dpi=300)
    rows = []
    entries = [("PI-BNN, fine-tuned", d, "\\textbf{PI-BNN}, S4 + adapted"), ("Deep ens. MLP, fine-tuned", d, "MLP, S4 + adapted"),
               ("PI-BNN, fine-tuned", w, "PI-BNN, wide + adapted"), ("Deep ens. MLP, fine-tuned", w, "MLP, wide + adapted"),
               ("PI-BNN, trained on real", d, "PI-BNN, real only"), ("Deep ens. MLP, trained on real", d, "MLP, real only"),
               ("ANN+GPR, trained on real", d, "ANN+GPR \\cite{farid2022}, real only"), ("Miner-RT, recalibrated on real", d, "Miner-RT, recalibrated"),
               ("PI-BNN, fine-tuned", load("crack_S4"), "\\textbf{PI-BNN}, S4 + adapted + inspections"),
               ("Deep ens. MLP, fine-tuned", load("crack_S4"), "MLP, S4 + adapted + inspections")]
    for m, dd, lab in entries:
        cells = []
        for k in KS:
            vals = {mm: stat(dd[(mm, k)], "CRPS")[0] for mm, _, _ in entries if (mm, k) in dd} if False else None
            c = stat(dd[(m, k)], "CRPS")[0] if (m, k) in dd else np.nan
            cells.append(f"{c:.2f}")
        cv = stat(dd[(m + " + CP", 0)], "PICP90")[0]
        rows.append(f"{lab} & " + " & ".join(cells) + f" & {cv:.2f} \\\\")
    zs = json.load(open(R + "real_summary.json"))["methods"]
    zero = f"PI-BNN, S4, no adaptation & \\multicolumn{{5}}{{c}}{{{zs['PI-BNN (proposed)']['CRPS'][0]:.2f}}} & {zs['PI-BNN (proposed) + CP(real)']['PICP90'][0]:.2f} \\\\\nMLP, S4, no adaptation & \\multicolumn{{5}}{{c}}{{{zs['Deep ensemble MLP']['CRPS'][0]:.2f}}} & {zs['Deep ensemble MLP + CP(real)']['PICP90'][0]:.2f} \\\\"
    tab = r"""\begin{table}[!htbp]
\centering
\caption{Adaptation to the SAE keyhole tests: CRPS of log RUL against the number $n_{\mathrm{t}}$ of real tests used, leave-one-group-out (each history and steel group is predicted by a model that saw only tests of the other five groups; ``all'' is about 39 tests). Mean over 5 seeds. The last column is the 90\% coverage with all tests, after cross-conformal scaling estimated within the training groups. ``S4'' and ``wide'': pretraining on the benchmark scenario S4 and on its domain-randomized variant. ``+ inspections'': the crack length measured so far is an additional real-time input. Rows without adaptation: CRPS as in Table~\ref{tab:real}, coverage after its leave-one-group-out conformal scaling.}
\label{tab:ft}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{5}{c}{CRPS with $n_{\mathrm{t}}$ real tests} & PICP90 \\
\cmidrule(lr){2-6}
Model & 3 & 5 & 10 & 20 & all & all \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + zero + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(OUT + "tab_ft.tex", "w").write(tab)
    mac = {}
    for (m, dd, key) in [("PI-BNN, fine-tuned", d, "Pft"), ("Deep ens. MLP, fine-tuned", d, "Mft"), ("PI-BNN, fine-tuned", w, "Pwide"),
                         ("Deep ens. MLP, fine-tuned", w, "Mwide"), ("PI-BNN, trained on real", d, "Preal"),
                         ("Deep ens. MLP, trained on real", d, "Mreal"), ("Miner-RT, recalibrated on real", d, "Mincal"),
                         ("ANN+GPR, trained on real", d, "Fareal")]:
        for k, kk in [(3, "Three"), (5, "Five"), (10, "Ten"), (20, "Twenty"), (0, "All")]:
            if (m, k) in dd:
                mac[f"ft{key}{kk}"] = f"{stat(dd[(m, k)], 'CRPS')[0]:.2f}"
                mac[f"ft{key}{kk}Cov"] = f"{stat(dd[(m + ' + CP', k)], 'PICP90')[0]:.2f}"
                mac[f"ft{key}{kk}Male"] = f"{stat(dd[(m, k)], 'MALE')[0]:.2f}"
                mac[f"ft{key}{kk}Sd"] = f"{stat(dd[(m, k)], 'CRPS')[1]:.2f}"
    with open(OUT + "numbers_ft.tex", "w") as f:
        for k, v in mac.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return mac


def crack_outputs():
    """Adds the crack-inspection (real-time health data) variant: macros, a real-time figure."""
    import pickle, sys
    sys.path.insert(0, "/home/claude/paper/realdata")
    from crack_obs import crack_records
    d = load("crack_S4")
    base = load("S4")
    mac = {}
    for m, key in [("PI-BNN, fine-tuned", "Pcr"), ("Deep ens. MLP, fine-tuned", "Mcr")]:
        for k, kk in [(3, "Three"), (5, "Five"), (10, "Ten"), (20, "Twenty"), (0, "All")]:
            if (m, k) in d:
                mac[f"ft{key}{kk}"] = f"{stat(d[(m, k)], 'CRPS')[0]:.2f}"
                mac[f"ft{key}{kk}Cov"] = f"{stat(d[(m + ' + CP', k)], 'PICP90')[0]:.2f}"
    # CRPS after crack detection vs before, all tests, seed-averaged (from prediction files)
    import metrics as Me
    rec = crack_records()
    ids = pickle.load(open(R + "pred_REAL_S4_s0.pkl", "rb"))["ids"]
    res = {}
    for tag, lab in [("", "load"), ("_crack", "crack")]:
        vals = {"pre": [], "post": []}
        for s in range(5):
            J = json.load(open(R + f"FT{tag}_S4_k0_s{s}.json"))
            idx = np.array(J["idx"]); tt = np.array(J["c"]["tt"]); rul = np.array(J["c"]["rul"])
            P = pickle.load(open(R + f"ftpred{tag}_S4_k0_s{s}_PI-BNN_fine-tuned.pkl", "rb"))
            det = np.array([(rec[ids[i]][0] <= t + 1e-9).any() and rec[ids[i]][1][rec[ids[i]][0] <= t + 1e-9][-1] > 0
                            for i, t in zip(idx, tt)])
            cr = Me.crps_samples(np.log(P["scp"].astype(float)), np.log(rul))
            vals["pre"].append(cr[~det].mean()); vals["post"].append(cr[det].mean())
        res[lab] = {k: float(np.mean(v)) for k, v in vals.items()}
    mac.update(crPreLoad=f"{res['load']['pre']:.2f}", crPostLoad=f"{res['load']['post']:.2f}",
               crPreCrack=f"{res['crack']['pre']:.2f}", crPostCrack=f"{res['crack']['post']:.2f}")
    with open(OUT + "numbers_crack.tex", "w") as f:
        for k, v in mac.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # real-time figure: load-only vs load + inspections, with the inspection record
    J = json.load(open(R + "FT_crack_S4_k0_s0.json"))
    idx = np.array(J["idx"]); tt = np.array(J["c"]["tt"]); rul = np.array(J["c"]["rul"]); Kf = np.array(J["c"]["Kf"])
    Pc = pickle.load(open(R + "ftpred_crack_S4_k0_s0_PI-BNN_fine-tuned.pkl", "rb"))
    Pl = pickle.load(open(R + "ftpred_S4_k0_s0_PI-BNN_fine-tuned.pkl", "rb"))
    show = ["SM3-2", "BM3-1", "TR2-2", "SR1-3"]
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.35))
    for ax, sid in zip(axs, show):
        j = ids.index(sid); m = idx == j; o = np.argsort(tt[m]); f = (tt[m] / Kf[m])[o]
        ax.plot(f, rul[m][o], color="k", lw=1.1, label="true RUL")
        lo, hi = np.quantile(Pc["scp"][m][o], [0.05, 0.95], 1)
        ax.fill_between(f, lo, hi, color="#d62728", alpha=0.16, lw=0, label="90% PI, load + inspections")
        ax.plot(f, Pc["pt"][m][o], color="#d62728", lw=1.1, label="PI-BNN, load + inspections")
        ax.plot(f, Pl["pt"][m][o], color="#d62728", lw=0.8, ls="--", label="PI-BNN, load only")
        ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_xlabel("life fraction")
        ax.set_title(f"{sid}, $K_f$ = {Kf[m][0]:,.0f}", fontsize=7.2)
        n, L = rec[sid]; ax2 = ax.twinx()
        ax2.plot(n / Kf[m][0], L, "o", ms=1.8, color="#7570b3", label="inspected crack length")
        ax2.set_ylim(0, 60); ax2.tick_params(labelsize=6, colors="#7570b3")
        if sid == show[-1]: ax2.set_ylabel("crack length (mm)", color="#7570b3", fontsize=6.5)
    axs[0].set_ylabel("RUL (blocks)")
    h, l = axs[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, fontsize=6, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT + "figs/fig_ft_examples.pdf"); fig.savefig("/home/claude/paper/figs/fig_ft_examples.png", dpi=300)
    return mac
