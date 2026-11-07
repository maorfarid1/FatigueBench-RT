"""Notch local-strain life model for the SAE keyhole specimen, with no parameter fitted to keyhole tests.

Notch elastic stress per unit load from the FE analysis of M. Groper (SolidWorks Simulation 2020, half
model, parabolic tetrahedra, 0.1 mm at the notch, 433,235 nodes): nominal stress S = P/A + M z/I =
11.265 P [MPa, P in kN] and K_t = 312.6/100.2 = 3.12. Cyclic stress-strain and strain-life curves
from the SAE FD&E fitted smooth-specimen data of the same steels (fde.uwaterloo.ca). Neuber's rule
for the notch root, Masing hysteresis per rainflow cycle and the Smith-Watson-Topper parameter.
"""
import json, sys, numpy as np
from scipy.optimize import brentq, least_squares
sys.path.insert(0, "/home/claude/paper/code")
import fatigue_physics as fp

S_PER_KN = 11.265        # nominal stress per kN (Groper FEA report, eq. 1)
KT = 3.12                # FEA stress concentration factor (312.6 / 100.217)
TAB = json.load(open("/home/claude/paper/realdata/strain_life_fitted.json"))


def fit_material(m):
    t = TAB[m]; E = t["E"]
    ea, N2, sa = map(np.array, (t["eps_a"], t["twoNf"], t["sig_a"]))
    keep = N2 < 1e7                       # drop the flat endurance points
    ea, N2, sa = ea[keep], N2[keep], sa[keep]
    ep = np.maximum(ea - sa / E, 1e-6)
    nprime, logK = np.polyfit(np.log(ep), np.log(sa), 1)
    def res(p):
        sf, b, ef, c = np.exp(p[0]), p[1], np.exp(p[2]), p[3]
        return np.log(sf / E * N2 ** b + ef * N2 ** c) - np.log(ea)
    p = least_squares(res, [np.log(900), -0.09, np.log(0.3), -0.5]).x
    return dict(E=E, K=float(np.exp(logK)), n=float(nprime), sf=float(np.exp(p[0])), b=float(p[1]),
                ef=float(np.exp(p[2])), c=float(p[3]), rms_fit=float(np.sqrt(np.mean(res(p) ** 2))))


MAT = {m: fit_material(m) for m in "MR"}


def neuber_amp(Se_a, k):
    """Local stress and strain amplitude from the elastic notch stress amplitude (Neuber + Masing)."""
    target = Se_a ** 2 / k["E"]
    f = lambda s: s * (s / k["E"] + (s / k["K"]) ** (1 / k["n"])) - target
    s = brentq(f, 1e-6, max(Se_a, 1.0) * 1.0001)
    return s, s / k["E"] + (s / k["K"]) ** (1 / k["n"])


def life_swt(swt, k):
    """2N_f from the SWT parameter sigma_max * eps_a (Newton on the log form)."""
    g = lambda x: np.log(k["sf"] ** 2 / k["E"] * x ** (2 * k["b"]) + k["sf"] * k["ef"] * x ** (k["b"] + k["c"])) - np.log(swt)
    lo, hi = 0.5, 1e12
    if g(hi) > 0: return hi
    if g(lo) < 0: return lo
    return brentq(lambda y: g(np.exp(y)), np.log(lo), np.log(hi))


def _life_lookup(k):
    """Vectorised SWT inversion on a log grid."""
    x = np.logspace(np.log10(0.5), 12, 4000)
    swt = k["sf"] ** 2 / k["E"] * x ** (2 * k["b"]) + k["sf"] * k["ef"] * x ** (k["b"] + k["c"])
    return np.log(swt[::-1]), np.log(x[::-1])


def _amp_lookup(k):
    s = np.logspace(-1, np.log10(5000), 4000)
    e = s / k["E"] + (s / k["K"]) ** (1 / k["n"])
    return np.log(s * e * k["E"]), np.log(s), np.log(e)    # log(Se^2), log sigma, log eps


LK = {m: (_life_lookup(MAT[m]), _amp_lookup(MAT[m])) for m in "MR"}


def damage_cycles(Pa_kN, Pmax_kN, cnt, m):
    """SWT damage of rainflow cycles given nominal load amplitude and max (kN)."""
    (lswt, lx), (lse2, ls, le) = LK[m]
    k = MAT[m]
    Sa = KT * S_PER_KN * np.abs(Pa_kN)
    Smax = KT * S_PER_KN * Pmax_kN
    ea = np.exp(np.interp(np.log(np.maximum(Sa, 1e-3) ** 2), lse2, le))
    smax = np.sign(Smax) * np.exp(np.interp(np.log(np.maximum(np.abs(Smax), 1e-3) ** 2), lse2, ls))
    swt = smax * ea
    ok = swt > 1e-6
    n2 = np.exp(np.interp(np.log(np.where(ok, swt, 1.0)), lswt, lx))
    return float(np.sum(np.where(ok, cnt / (n2 / 2.0), 0.0)))


def block_damage_load(P_N, m):
    """Damage of one pass of a repeated load history P (N)."""
    a, mean, n = fp.rainflow_cycles(np.concatenate([P_N, P_N]) / 1000.0)
    return damage_cycles(a, mean + a, n, m) / 2.0


if __name__ == "__main__":
    from parse_keyhole import history
    for m, k in MAT.items(): print(m, {a: round(v, 4) for a, v in k.items()})
    d = json.load(open("/home/claude/paper/realdata/keyhole_tests.json"))
    print("CA tests (no fitting):")
    rc = []
    for r in d["ca"]:
        Pa = (r["Pmax"] - r["Pmin"]) / 2000; Pm = (r["Pmax"] + r["Pmin"]) / 2000
        N = 1.0 / damage_cycles(np.array([Pa]), np.array([Pm + Pa]), np.array([1.0]), r["mat"])
        rc.append(np.log10(r["Nf_rev"] / 2 / N))
        print("  %-5s %s Pa %5.1f kN  obs %9.0f  pred %9.0f  ratio %.2f" % (r["id"], r["mat"], Pa, r["Nf_rev"] / 2, N, r["Nf_rev"] / 2 / N))
    rc = np.array(rc); print("  CA log10(obs/pred) mean %.2f sd %.2f" % (rc.mean(), rc.std(ddof=1)))
    H = {n: history(n) for n in ["bracket", "suspension", "transmission"]}
    rows = []
    for r in d["va"]:
        x = H[r["history"]]; ext = x.min() if r["Pext"] < 0 else x.max()
        P = x * abs(r["Pext"]) / abs(ext)
        L = 1.0 / block_damage_load(P, r["mat"])
        rows.append(dict(id=r["id"], censored=r["censored"], K_f=r["K_f"], K_ls=L))
        print("  %-6s %s Kf %9.1f  pred %10.1f  ratio %.2f" % (r["id"], "C" if r["censored"] else " ", r["K_f"], L, r["K_f"] / L))
    rt = np.log10([q["K_f"] / q["K_ls"] for q in rows if not q["censored"]])
    print("VA failures (no fitting): log10(obs/pred) mean %.2f sd %.2f; within x2 %.2f; within x3 %.2f"
          % (rt.mean(), rt.std(ddof=1), np.mean(np.abs(rt) < np.log10(2)), np.mean(np.abs(rt) < np.log10(3))))
    json.dump(dict(mat=MAT, ca_log10_ratio=rc.tolist(), va=rows), open("/home/claude/paper/realdata/local_strain_va.json", "w"), indent=1)
