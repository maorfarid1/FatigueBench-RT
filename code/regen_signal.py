"""Regenerate the measured stress windows of one benchmark component from its seed (exact replay of gen.py)."""
import inspect, numpy as np, pickle, gen
src = inspect.getsource(gen._simulate)
src = src.replace("def _simulate(scen, seed, variant):", "def _simulate_x(scen, seed, variant):\n    XS = []")
src = src.replace('        xm = xc + rng.normal(0, NOISE * amp, xc.size)\n', '        xm = xc + rng.normal(0, NOISE * amp, xc.size)\n        XS.append(xm.astype(np.float32))\n')
src = src.replace("    return out\n", "    out['windows'] = XS\n    return out\n")
ns = dict(vars(gen)); exec(src, ns)

def windows(traj):
    out = ns["_simulate_x"](traj["scen"], int(traj["seed"]), traj["variant"])
    assert np.allclose(out["d_true"], traj["d_true"]), "replay mismatch"
    return out["windows"]

if __name__ == "__main__":
    import sys
    t = pickle.load(open(f"/home/claude/paper/data/{sys.argv[1]}.pkl", "rb"))[int(sys.argv[2])]
    w = windows(t); print(len(w), w[0].shape, "replay OK")
