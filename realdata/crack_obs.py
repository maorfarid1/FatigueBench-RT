"""Crack length inspections recorded during the SAE keyhole VA tests: {specimen id: (blocks, length mm)}."""
import os, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))


def crack_records():
    cwd = os.getcwd(); os.chdir(_HERE)
    try:
        from parse_keyhole import tables
        out = {}
        for t in tables():
            parts = t["name"].split()
            if len(parts) < 2 or not parts[0].endswith("E"): continue
            sid = parts[1].upper().replace("-FS", "")
            if sid == "SR2-2" and np.nanmax(t["n"]) > 10000: sid = "SR3-2"
            rows = t["rows"]
            n = pd.to_numeric(rows[1], errors="coerce").values
            L6 = pd.to_numeric(rows[6], errors="coerce").values if 6 in rows else np.full(len(n), np.nan)
            L2 = pd.to_numeric(rows[2], errors="coerce").values
            L = np.where(np.isfinite(L6) & (L6 >= 0), L6, L2)
            ok = np.isfinite(n) & np.isfinite(L) & (L >= 0)
            o = np.argsort(n[ok])
            out[sid] = (n[ok][o], L[ok][o])
        return out
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    r = crack_records()
    for k, (n, L) in r.items(): print(k, len(n), n[:3], L[:3], n[-1], L[-1])
