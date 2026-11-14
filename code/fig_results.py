"""Aggregate results into LaTeX tables and figures."""
import json, glob, pickle, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as Me

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "legend.frameon": False})
R = "/home/claude/paper/results/"
OUT = "/home/claude/paper/tex/"
F = "/home/claude/paper/figs/"
METHODS = ["Miner-RT (rainflow)", "Dirlik-RT (spectral)", "ANN+GPR (Farid 2022)", "ANN+GPR (Farid 2022) + CP",
           "Deep ensemble MLP", "Deep ensemble MLP + CP", "Deep ensemble LSTM", "Deep ensemble LSTM + CP",
           "PI-BNN (proposed)", "PI-BNN (proposed) + CP"]
SHORT = {"Miner-RT (rainflow)": "Miner-RT (rainflow)", "Dirlik-RT (spectral)": "Dirlik-RT (spectral)",
         "ANN+GPR (Farid 2022)": "ANN+GPR \\cite{farid2022}", "ANN+GPR (Farid 2022) + CP": "\\quad + CP",
         "Deep ensemble MLP": "Deep ens.\\ MLP", "Deep ensemble MLP + CP": "\\quad + CP",
         "Deep ensemble LSTM": "Deep ens.\\ LSTM", "Deep ensemble LSTM + CP": "\\quad + CP",
         "PI-BNN (proposed)": "\\textbf{PI-BNN (ours)}", "PI-BNN (proposed) + CP": "\\quad \\textbf{+ CP}"}
COL = {"Miner-RT (rainflow)": "#7f7f7f", "Dirlik-RT (spectral)": "#bcbd22", "ANN+GPR (Farid 2022)": "#9467bd",
       "Deep ensemble MLP": "#1f77b4", "Deep ensemble LSTM": "#17becf", "PI-BNN (proposed)": "#d62728"}
LAB = {"Miner-RT (rainflow)": "Miner-RT", "Dirlik-RT (spectral)": "Dirlik-RT", "ANN+GPR (Farid 2022)": "ANN+GPR [2022]",
       "Deep ensemble MLP": "Deep ens. MLP", "Deep ensemble LSTM": "Deep ens. LSTM", "PI-BNN (proposed)": "PI-BNN (ours)"}
KEYS = [("CRPS", "CRPS", 1, "{:.3f}"), ("MALE", "MALE", 1, "{:.3f}"), ("NRMSE", "NRMSE", 1, "{:.2f}"),
        ("alpha_lambda", "$\\alpha$-$\\lambda$", -1, "{:.2f}"), ("PICP90", "PICP90", 0, "{:.2f}"),
        ("MPIW90", "MPIW90", None, "{:.2f}"), ("ECE", "ECE", 1, "{:.3f}"), ("PH", "PH", -1, "{:.2f}")]


def load(pattern):
    out = {}
    for f in sorted(glob.glob(R + pattern)):
        d = json.load(open(f))
        for v, res in d["results"].items():
            for m, r in res.items():
                out.setdefault((v, m), []).append(r)
    return out


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return v.mean(), v.std(), len(v)


def fmt_num(x, f):
    if not np.isfinite(x): return "--"
    if abs(x) >= 100: return "$>$99"
    return f.format(x)


def table(blocks, caption, label, methods=METHODS, keys=KEYS):
    """blocks: list of (title, dict)."""
    cols = "l" + "c" * len(keys)
    s = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize",
         "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
         "Method & " + " & ".join(k[1] for k in keys) + " \\\\"]
    for title, res in blocks:
        s += ["\\midrule", f"\\multicolumn{{{len(keys)+1}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        present = [m for m in methods if ("base", m) in res or any(m == kk[1] for kk in res)]
        # best values (exclude methods without results)
        best = {}
        for key, _, sgn, _ in keys:
            vals = {m: agg(res[next(kk for kk in res if kk[1] == m)], key)[0] for m in present}
            if sgn is None: best[key] = None
            elif sgn == 1: best[key] = min(vals.values())
            elif sgn == -1: best[key] = max(vals.values())
            else: best[key] = min(vals.values(), key=lambda x: abs(x - 0.9)); best[key] = [m for m in vals if abs(vals[m] - 0.9) == abs(best[key] - 0.9)]
        for m in present:
            rows = res[next(kk for kk in res if kk[1] == m)]
            cells = []
            for key, _, sgn, f in keys:
                mu, sd, n = agg(rows, key)
                c = fmt_num(mu, f)
                isbest = False if sgn is None else ((m in best[key]) if sgn == 0 else np.isclose(mu, best[key]))
                if isbest and c not in ("--", "$>$99"): c = "\\textbf{" + c + "}"
                if n > 1 and sd >= 5e-4 and abs(mu) < 100 and key in ("CRPS", "MALE"):
                    c += "{\\tiny$\\pm$" + f.format(sd) + "}"
                cells.append(c)
            s.append(SHORT.get(m, m) + " & " + " & ".join(cells) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(s)


def subset(res, variant):
    return {k: v for k, v in res.items() if k[0] == variant}


def numbers():
    """Write LaTeX macros with the headline numbers used in the text."""
    st = json.load(open(R + "data_stats.json"))
    import pickle as _p
    n_extra = sum(len(_p.load(open(f"/home/claude/paper/data/S3_test_{v}.pkl", "rb"))) for v in ["sae", "ood_kurt"])
    ntraj = sum(st[s]["n"] for s in st) + n_extra
    nblk = sum(st[s]["nblk"] for s in st)
    mac = {"NTRAJ": f"{ntraj:,}", "NBLK": f"{nblk:,}"}
    for sc in ["S1", "S2", "S3", "S4"]:
        mac[f"ratio{sc}"] = f"{1 / st[sc]['ratio_med']:.0f}" if st[sc]["ratio_med"] < 1 else f"{st[sc]['ratio_med']:.2f}"
        mac[f"life{sc}"] = f"{st[sc]['life_med']:.0f}"
    def g(pattern, m, key, v="base"):
        res = subset(load(pattern), v)
        rows = res.get((v, m), [])
        return np.mean([r[key] for r in rows]) if rows else np.nan
    P, PC, MLP, MLPC, LS, MI, FA = ("PI-BNN (proposed)", "PI-BNN (proposed) + CP", "Deep ensemble MLP", "Deep ensemble MLP + CP",
                                    "Deep ensemble LSTM", "Miner-RT (rainflow)", "ANN+GPR (Farid 2022)")
    for sc in ["S1", "S2", "S3", "S4"]:
        for tag, m in [("P", PC), ("M", MLP), ("L", LS), ("Mi", MI), ("F", FA), ("Praw", P)]:
            mac[f"crps{tag}{sc}"] = f"{g(f'{sc}_iid_n600_s*.json', m, 'CRPS'):.3f}"
            mac[f"male{tag}{sc}"] = f"{g(f'{sc}_iid_n600_s*.json', m, 'MALE'):.2f}"
            mac[f"picp{tag}{sc}"] = f"{g(f'{sc}_iid_n600_s*.json', m, 'PICP90'):.2f}"
    for sc in ["S2", "S3", "S4"]:
        for tag, m in [("P", PC), ("M", MLP), ("L", LS), ("Mi", MI), ("F", FA)]:
            mac[f"xcrps{tag}{sc}"] = f"{g(f'{sc}_extrap_n*_s*.json', m, 'CRPS'):.2f}"
            mac[f"xpicp{tag}{sc}"] = f"{g(f'{sc}_extrap_n*_s*.json', m, 'PICP90'):.2f}"
            mac[f"xmale{tag}{sc}"] = f"{g(f'{sc}_extrap_n*_s*.json', m, 'MALE'):.2f}"
    with open(OUT + "numbers.tex", "w") as f:
        for k, v in mac.items():
            name = re.sub(r"\d", lambda d: "ABCDEFGHIJ"[int(d.group())], k)
            f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
    return mac


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["main", "extrap", "stress", "ablation", "eff", "figs", "numbers"]
    if "main" in what:
        blocks = []
        for sc, name in [("S1", "S1: stationary Gaussian"), ("S2", "S2: non-stationary mission mix"),
                         ("S3", "S3: non-stationary, non-Gaussian"), ("S4", "S4: drift and load-sequence effects")]:
            res = subset(load(f"{sc}_iid_n600_s*.json"), "base")
            if res: blocks.append((name, res))
        open(OUT + "tab_main.tex", "w").write(table(blocks,
            "In-distribution results on FatigueBench-RT (250 test components per scenario, mean over 3 seeds; $\\pm$ one standard deviation for CRPS and MALE). Lower is better except $\\alpha$--$\\lambda$ and PH (higher) and PICP90 (target 0.90). Best values in bold. ``+CP'': split-conformal calibration.",
            "tab:main"))
    if "extrap" in what:
        blocks = []
        for sc, name in [("S2", "S2 life extrapolation"), ("S3", "S3 life extrapolation"), ("S4", "S4 life extrapolation")]:
            res = subset(load(f"{sc}_extrap_n*_s*.json"), "base")
            if res: blocks.append((name, res))
        open(OUT + "tab_extrap.tex", "w").write(table(blocks,
            "Life-scale extrapolation: models trained and calibrated on components with $K_f<100$ blocks and tested on components with $K_f>150$ blocks (mean over 3 seeds).",
            "tab:extrap"))
    if "stress" in what:
        res = load("S3_iid_n600_s*.json")
        blocks = [("S3 test, real SAE steels (Man-Ten, RQC-100)", subset(res, "sae")),
                  ("S3 test, kurtosis extrapolation ($\\gamma_4\\in[7,10]$)", subset(res, "ood_kurt"))]
        open(OUT + "tab_stress.tex", "w").write(table(blocks,
            "Stress tests with S3-trained models: real SAE steels and kurtosis outside the training range (mean over 3 seeds).",
            "tab:stress"))
    if "ablation" in what:
        AB = ["PI-BNN (proposed)", "PI-BNN w/o physics priors", "PI-BNN w/o scale-free inputs", "PI-BNN w/o standard-error shrinkage",
              "PI-BNN w/o damage/rate supervision", "PI-BNN single network", "Deep ensemble MLP"]
        for k in AB: SHORT.setdefault(k, k)
        SHORT["PI-BNN (proposed)"] = "PI-BNN (full)"
        iid = subset(load("S3_iid_n600_s*.json"), "base")
        iid.update(subset(load("S3abl_iid_n600_s*.json"), "base"))
        ex = subset(load("S3_extrap_n*_s*.json"), "base")
        keys = [KEYS[0], KEYS[1], KEYS[4], KEYS[6]]
        open(OUT + "tab_ablation.tex", "w").write(table([("S3 in-distribution", iid), ("S3 life extrapolation", ex)],
            "Ablation of PI-BNN components on S3 (mean over 3 seeds, raw predictive distributions without conformal calibration).",
            "tab:ablation", methods=AB, keys=keys).replace("table*", "table").replace("\\begin{table}[t]", "\\begin{table}[t]"))
        SHORT["PI-BNN (proposed)"] = "\\textbf{PI-BNN (ours)}"
    if "eff" in what:
        fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
        for j, sc in enumerate(["S3", "S4"]):
            for m in ["Miner-RT (rainflow)", "ANN+GPR (Farid 2022)", "Deep ensemble MLP", "PI-BNN (proposed)"]:
                ns, mu, sd, cov = [], [], [], []
                for n in [25, 50, 100, 200, 400, 600]:
                    res = subset(load(f"{sc}_iid_n{n}_s*.json"), "base")
                    if ("base", m) in res:
                        mk = m + " + CP" if ("base", m + " + CP") in res else m
                        v = np.array([r["CRPS"] for r in res[("base", mk)]])
                        ns.append(n); mu.append(v.mean()); sd.append(v.std())
                        cov.append(np.mean([r["PICP90"] for r in res[("base", m + " + CP")]]) if ("base", m + " + CP") in res else np.mean([r["PICP90"] for r in res[("base", m)]]))
                if ns:
                    ns, mu, sd = map(np.array, (ns, mu, sd))
                    axs[2 * j].plot(ns, mu, "o-", ms=2.5, lw=1, color=COL[m], label=LAB[m])
                    axs[2 * j].fill_between(ns, mu - sd, mu + sd, color=COL[m], alpha=0.15, lw=0)
                    axs[2 * j + 1].plot(ns, cov, "o-", ms=2.5, lw=1, color=COL[m])
            for a_ in (axs[2 * j], axs[2 * j + 1]):
                a_.set_xscale("log"); a_.set_xticks([25, 100, 600]); a_.set_xticklabels(["25", "100", "600"]); a_.minorticks_off()
            axs[2 * j].set_title(f"{sc}: CRPS of log RUL"); axs[2 * j].set_xlabel("training components")
            axs[2 * j + 1].axhline(0.9, color="0.5", ls="--", lw=0.6)
            axs[2 * j + 1].set_title(f"{sc}: PICP90 (after CP)"); axs[2 * j + 1].set_xlabel("training components")
        axs[0].legend(fontsize=5.8)
        plt.tight_layout(); plt.savefig(F + "fig8_efficiency.pdf"); plt.savefig(F + "fig8_efficiency.png", dpi=200)
    if "figs" in what:
        # Fig 4 example real-time predictions, Fig 5 by life fraction, Fig 6 calibration
        show = ["Miner-RT (rainflow)", "ANN+GPR (Farid 2022)", "Deep ensemble MLP", "PI-BNN (proposed)"]
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
        for j, sc in enumerate(["S2", "S3", "S4"]):
            P = pickle.load(open(R + f"pred_{sc}_iid_n600_s0_base.pkl", "rb"))
            c, idx = P["c"], P["idx"]
            ids = np.unique(idx)
            Kf = np.array([c["Kf"][idx == i][0] for i in ids])
            ptp = P["preds"]["PI-BNN (proposed) + CP"][0]
            err = np.array([np.mean(np.abs(np.log(ptp[idx == i]) - np.log(c["rul"][idx == i]))) for i in ids])
            cand = np.where((Kf > 80) & (Kf < 160))[0]
            i = ids[cand[np.argsort(np.abs(err[cand] - np.median(err[cand])))[0]]]   # typical component
            m = idx == i
            t = c["tt"][m]; ax = axs[j]
            ax.plot(t, c["rul"][m], "k-", lw=1.2, label="true RUL")
            for meth in show:
                pt, smp = P["preds"][meth + (" + CP" if meth == "PI-BNN (proposed)" else "")]
                if meth == "PI-BNN (proposed)":
                    lo, hi = np.quantile(smp[m], [0.05, 0.95], axis=1)
                    ax.fill_between(t, lo, hi, color=COL[meth], alpha=0.18, lw=0, label="PI-BNN 90% PI")
                ax.plot(t, pt[m], color=COL[meth], lw=1.2 if meth == "PI-BNN (proposed)" else 0.7,
                        alpha=1.0 if meth == "PI-BNN (proposed)" else 0.8, label=LAB[meth])
            ax.set_ylim(0, 2.2 * c["Kf"][m][0]); ax.set_xlim(0, c["Kf"][m][0])
            ax.set_title(f"{sc}, typical component, $K_f$={c['Kf'][m][0]:.0f}", fontsize=7); ax.set_xlabel("operating time $t$ (blocks)")
            if j == 0: ax.set_ylabel("RUL (blocks)"); ax.legend(fontsize=5.6, loc="upper right")
        plt.tight_layout(); plt.savefig(F + "fig4_examples.pdf"); plt.savefig(F + "fig4_examples.png", dpi=200)

        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
        P = pickle.load(open(R + "pred_S3_iid_n600_s0_base.pkl", "rb"))
        for meth in ["Miner-RT (rainflow)", "Dirlik-RT (spectral)", "ANN+GPR (Farid 2022)", "Deep ensemble MLP", "Deep ensemble LSTM", "PI-BNN (proposed)"]:
            key = meth if meth in ("Miner-RT (rainflow)", "Dirlik-RT (spectral)") else meth + " + CP"
            if key not in P["preds"]: continue
            pt, smp = P["preds"][key]
            frac = P["c"]["tt"] / P["c"]["Kf"]
            bins = np.linspace(0.05, 0.95, 10); xs, cr, cov = [], [], []
            crps = Me.crps_samples(np.log(smp), np.log(P["c"]["rul"]))
            male = np.abs(np.log(pt) - np.log(P["c"]["rul"]))
            lo, hi = np.quantile(smp, [0.05, 0.95], axis=1)
            inside = (P["c"]["rul"] >= lo) & (P["c"]["rul"] <= hi)
            ml = []
            for a, b in zip(bins[:-1], bins[1:]):
                s = (frac >= a) & (frac < b); xs.append((a + b) / 2); cr.append(np.mean(crps[s])); cov.append(np.mean(inside[s])); ml.append(np.mean(male[s]))
            axs[0].plot(xs, cr, "o-", ms=2.5, lw=1, color=COL[meth], label=LAB[meth])
            axs[1].plot(xs, ml, "o-", ms=2.5, lw=1, color=COL[meth])
            axs[2].plot(xs, cov, "o-", ms=2.5, lw=1, color=COL[meth])
        axs[0].set_ylabel("CRPS of log RUL"); axs[0].set_yscale("log"); axs[1].set_ylabel("MALE"); axs[1].set_yscale("log")
        axs[2].axhline(0.9, color="0.5", ls="--", lw=0.6); axs[2].set_ylabel("PICP90"); axs[2].set_ylim(0, 1.02)
        for a in axs: a.set_xlabel("life fraction $t/K_f$")
        axs[0].legend(fontsize=5.6)
        plt.tight_layout(); plt.savefig(F + "fig5_lifefraction.pdf"); plt.savefig(F + "fig5_lifefraction.png", dpi=200)

        fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
        for j, sc in enumerate(["S1", "S2", "S3", "S4"]):
            P = pickle.load(open(R + f"pred_{sc}_iid_n600_s0_base.pkl", "rb"))
            frac = P["c"]["tt"] / P["c"]["Kf"]; s = (frac >= 0.1) & (frac <= 0.95)
            for meth in ["Miner-RT (rainflow)", "ANN+GPR (Farid 2022)", "Deep ensemble MLP", "Deep ensemble LSTM", "PI-BNN (proposed)"]:
                if meth not in P["preds"]: continue
                pt, smp = P["preds"][meth]
                lv = np.linspace(0.1, 0.9, 9)
                cov = [np.mean((P["c"]["rul"][s] >= np.quantile(smp[s], (1 - a) / 2, axis=1)) & (P["c"]["rul"][s] <= np.quantile(smp[s], (1 + a) / 2, axis=1))) for a in lv]
                axs[j].plot(lv, cov, "o-", ms=2, lw=0.9, color=COL[meth], label=LAB[meth])
            axs[j].plot([0, 1], [0, 1], "k--", lw=0.6); axs[j].set_title(sc + " (before CP)")
            axs[j].set_xlabel("nominal coverage")
            if j == 0: axs[j].set_ylabel("observed coverage")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(F + "fig6_calibration.pdf"); plt.savefig(F + "fig6_calibration.png", dpi=200)
    if "numbers" in what:
        print(numbers())
    print("done", what)
