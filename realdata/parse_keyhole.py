"""Parse the public SAE FD&E keyhole test program (Tucker & Bussa 1975) into run-to-failure records.

Sources (GPL, SAE FD&E committee, hosted at fde.uwaterloo.ca/Fde/Loads/Keyhole/):
  bracket.txt, suspension.txt, transmission.txt  : normalized load histories (peak-valley, +-999 = extreme)
  keyholeCrackPropData.ods                        : CA and VA test tables (crack length vs reversals/blocks)
  keyholeSpec.Histories.pdf                       : load levels per specimen (plot labels)
"""
import json, numpy as np, pandas as pd

def history(name):
    x = np.array([float(l.split(":")[0]) for l in open(f"{name}.txt") if l.strip() and not l.startswith("#")])
    return x / 999.0

# extreme load of each VA load level [N]; sign = sign of the history extreme the level refers to
LEVELS = {
    ("bracket", 1): -71200, ("bracket", 2): -35600, ("bracket", 3): -15600, ("bracket", 4): -13300, ("bracket", 5): -11100,
    ("suspension", 1): -71200, ("suspension", 2): -40000, ("suspension", 4): -20000, ("suspension", 5): -13300,
    ("transmission", 1): 71200, ("transmission", 2): 35600, ("transmission", 3): 15600,
}
SUSP = {("M", 3): -26700, ("R", 3): -31100, ("R", 4): -26700}   # suspension levels 3-4 differ between the steels (PDF labels)
HIST = {"B": "bracket", "S": "suspension", "T": "transmission"}
# specimens that did not fail (suspended or run-out), from the table notes and the PDF labels
CENSORED = {"SM4-3", "SM5-1", "BM4-2", "BM5-1", "SR2-1", "SR2-2", "SR4-1"}

def tables():
    v = pd.read_excel("keyholeCrackPropData.ods", engine="odf", header=None)
    lab = [str(x) for x in v[0]]
    st = [i for i in range(len(v)) if "Table" in lab[i]]
    out = []
    for j, s in enumerate(st):
        e = st[j + 1] if j + 1 < len(st) else len(v)
        name = lab[s].replace("Table", "").strip()
        n = pd.to_numeric(v.loc[s:e - 1, 1], errors="coerce")
        L = pd.to_numeric(v.loc[s:e - 1, 2], errors="coerce")
        out.append(dict(name=name, rows=v.loc[s:e - 1], n=n.values, L=L.values))
    return out

def ca_tests():
    """Constant amplitude tests: (material, Pmax, Pmin, reversals to failure)."""
    res = []
    for t in tables():
        if "CA" not in t["name"]:
            continue
        num = int("".join(c for c in t["name"].split()[0] if c.isdigit()))
        mat = "M" if num <= 6 else "R"
        P = pd.to_numeric(t["rows"][6], errors="coerce").dropna().values
        Q = pd.to_numeric(t["rows"][7], errors="coerce").dropna().values
        n = t["n"][np.isfinite(t["n"])]
        res.append(dict(id=t["name"].split("CA")[-1].strip(), mat=mat, Pmax=float(P[0]), Pmin=float(Q[0]),
                        Nf_rev=float(np.nanmax(n))))
    return res

def va_tests():
    res = []
    for t in tables():
        if not t["name"].endswith(("E", "-FS")) and "E " not in t["name"]:
            pass
        parts = t["name"].split()
        if len(parts) < 2 or not parts[0].endswith("E"):
            continue
        sid = parts[1].upper().replace("-FS", "")
        if sid == "SR2-2" and np.nanmax(t["n"]) > 10000:   # mislabelled duplicate in the sheet; PDF: SR3-2, -31100 N
            sid = "SR3-2"
        h, m, lvl = HIST[sid[0]], sid[1], int(sid[2])
        P = SUSP[(m, lvl)] if (m, lvl) in SUSP and h == "suspension" else LEVELS[(h, lvl)]
        n, L = t["n"], t["L"]
        ok = np.isfinite(n)
        n_obs, L_obs = n[ok], L[ok]
        init = n_obs[np.nanargmax(np.nan_to_num(L_obs, nan=-1) >= 2.54)] if np.any(np.nan_to_num(L_obs, nan=-1) >= 2.54) else None
        res.append(dict(id=sid, history=h, mat=m, level=lvl, Pext=float(P), K_f=float(np.nanmax(n_obs)),
                        K_init=None if init is None else float(init), censored=sid in CENSORED))
    return res

if __name__ == "__main__":
    ca, va = ca_tests(), va_tests()
    json.dump(dict(ca=ca, va=va), open("keyhole_tests.json", "w"), indent=1)
    for r in ca: print("CA", r)
    for r in va: print("VA", r)
    print(len(ca), "CA tests;", len(va), "VA tests;", sum(not r["censored"] for r in va), "VA failures")
