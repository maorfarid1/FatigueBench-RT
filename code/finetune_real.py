"""Fine-tuning on measured tests: pretrain on simulated data, adapt to the SAE keyhole tests of the
other history/steel groups, predict the held-out group (leave-one-group-out, 6 groups).

Every adapted model sees no data of the group it is scored on. Prediction intervals are optionally
rescaled with a cross-conformal factor estimated by an inner leave-one-group-out loop on the
training groups only.

python finetune_real.py --pre S4 --seed 0          (pretraining set S4 or S4_wide)
"""
import argparse, copy, json, pickle, time, numpy as np, torch
import features as F, models as Mo, metrics as Me
from scipy.stats import norm

p = argparse.ArgumentParser()
p.add_argument("--pre", default="S4"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--ft_epochs", type=int, default=100); p.add_argument("--ft_lr", type=float, default=5e-4)
p.add_argument("--no_inner", action="store_true")
p.add_argument("--k", type=int, default=0, help="fine-tune on only k random specimens of the other groups (0 = all)")
p.add_argument("--methods", default="")
a = p.parse_args()
torch.manual_seed(a.seed)
D = "/home/claude/paper/data/"
ld = lambda n: pickle.load(open(D + n + ".pkl", "rb"))
sfx = "_wide" if a.pre.endswith("_wide") else ""
tr, va, te = ld(f"S4_train{sfx}"), ld(f"S4_val{sfx}"), ld("REAL_keyhole")
rng = np.random.default_rng(a.seed)
EP = int(np.clip(60 * np.sqrt(100 / len(tr)), 20, 120))
Str, Sva = Mo.stack(F.build(tr)), Mo.stack(F.build(va))
Ste = Mo.stack(F.build(te))
for S_ in (Str, Sva, Ste):
    S_[1]["dD"] = np.maximum(S_[1]["dD"], 1e-9); S_[1]["rate_future"] = np.maximum(S_[1]["rate_future"], 1e-12)

# evaluation grid (identical to run_real.py)
c, idx = Ste[1], Ste[2]
keep = np.zeros(len(idx), bool)
for j in np.unique(idx):
    m = np.where(idx == j)[0]
    if len(m) <= 40: keep[m] = True
    else:
        for f in np.arange(0.05, 0.951, 0.05): keep[m[np.argmin(np.abs(c["tt"][m] - f * c["Kf"][m][0]))]] = True
R = (Ste[0][keep], {k: v[keep] for k, v in c.items()}, idx[keep])
ids = [t["id"] for t in te]
G = np.array([ids[i][0] + ids[i][1] for i in R[2]])          # history x steel group of each row
GROUPS = sorted(set(G))


def sub(mask):
    X, cc, ii = R[0][mask], {k: v[mask] for k, v in R[1].items()}, R[2][mask]
    cnt = {j: (ii == j).sum() for j in np.unique(ii)}
    cc["w"] = np.array([1.0 / cnt[j] for j in ii])          # every specimen weighs the same
    return X, cc, ii


T = lambda x: torch.tensor(x, dtype=torch.float32)


def ft_mlp(base, S):
    M = copy.deepcopy(base); X, cc, _ = S
    Xt, y, w = T(M.sc(X)), T(np.log(cc["rul"])), T(cc["w"] / cc["w"].mean())
    for k, net in enumerate(M.nets):
        def loss(b, net=net):
            o = net(Xt[b]); return (w[b] * Mo.gnll(y[b], o[:, 0] + M.ym, o[:, 1])).mean()
        Mo.train_loop(net, loss, len(y), a.ft_epochs, a.seed * 10 + k, bs=64, lr=a.ft_lr)
    return M


def ft_pibnn(base, S):
    M = copy.deepcopy(base); X, cc, _ = S
    Xt, w = T(M.sc(M._inp(X))), T(cc["w"] / cc["w"].mean())
    pD, pr, se = map(T, M._priors(cc))
    yD, yr, y = T(np.log(cc["dD"])), T(np.log(cc["rate_future"])), T(np.log(cc["rul"]))
    for k, net in enumerate(M.nets):
        def loss(b, net=net):
            muD, lvD, mur, lvr = M._heads(net(Xt[b]), pD[b], pr[b], se[b])
            L = Mo.gnll(y[b], muD - mur, torch.logaddexp(lvD, lvr))
            if M.aux: L = L + M.lam * (Mo.gnll(yD[b], muD, lvD) + Mo.gnll(yr[b], mur, lvr))
            return (w[b] * L).mean()
        Mo.train_loop(net, loss, len(y), a.ft_epochs, a.seed * 10 + k, bs=64, lr=a.ft_lr)
    return M


class MinerCal:
    """Real-time Miner extrapolation with bias and scatter re-estimated on real tests."""
    name = "Miner-RT, recalibrated"
    def fit(self, S):
        self.base = Mo.MinerExtrapolation(); r = np.log(S[1]["rul"]) - self.base._mu(S[1])
        self.bias, self.sd = float(np.average(r, weights=S[1]["w"])), float(np.sqrt(np.average((r - np.average(r, weights=S[1]["w"])) ** 2, weights=S[1]["w"])))
        return self
    def predict(self, S, rng):
        mu = (self.base._mu(S[1]) + self.bias)[:, None]
        return np.exp(mu[:, 0]), Mo.mixture_samples(mu, np.full_like(mu, self.sd), rng)


# pretrained (simulation) models for this seed
t0 = time.time()
PRE = {"mlp": Mo.DeepEnsembleMLP(epochs=EP).fit(Str, Sva, a.seed), "pibnn": Mo.PIBNN(epochs=EP).fit(Str, Sva, a.seed)}
print(f"pretrained on {a.pre} in {time.time() - t0:.0f}s", flush=True)

METHODS = {
    "Miner-RT, recalibrated on real": lambda S: MinerCal().fit(S),
    "ANN+GPR, trained on real": lambda S: Mo.Farid2022().fit(S, S, a.seed),
    "Deep ens. MLP, trained on real": lambda S: Mo.DeepEnsembleMLP(epochs=150).fit(S, S, a.seed),
    "PI-BNN, trained on real": lambda S: Mo.PIBNN(epochs=150).fit(S, S, a.seed),
    "Deep ens. MLP, fine-tuned": lambda S: ft_mlp(PRE["mlp"], S),
    "PI-BNN, fine-tuned": lambda S: ft_pibnn(PRE["pibnn"], S),
}


def limit_k(train_mask, key):
    if not a.k: return train_mask
    spec = np.unique(R[2][train_mask])
    rs = np.random.default_rng(1000 * a.seed + key)
    pick = rs.choice(spec, size=min(a.k, len(spec)), replace=False)
    return train_mask & np.isin(R[2], pick)


def fit_predict(name, train_mask, test_mask, key=0):
    train_mask = limit_k(train_mask, key)
    M = METHODS[name](sub(train_mask))
    P = M.predict(sub(test_mask), rng)
    return np.median(P[1], 1), P[1]


z = norm.ppf(0.95)
out = {}
for name in [m for m in METHODS if not a.methods or any(x in m for x in a.methods.split(";"))]:
    t0 = time.time()
    pt = np.zeros(len(G)); S_ = np.zeros((len(G), Mo.S_SAMPLES)); Scp = np.zeros_like(S_); lams = []
    for g in GROUPS:
        trm, tem = G != g, G == g
        pg, sg = fit_predict(name, trm, tem, key=GROUPS.index(g))
        pt[tem], S_[tem] = pg, sg
        lam = 1.0
        if not a.no_inner:      # inner leave-one-group-out on the training groups only
            scores = []
            for h in GROUPS:
                if h == g: continue
                ph, sh = fit_predict(name, trm & (G != h), G == h, key=100 + 10 * GROUPS.index(g) + GROUPS.index(h))
                lm, ls = np.log(np.median(sh, 1)), np.std(np.log(sh), 1) + 1e-9
                scores.append(np.abs(np.log(R[1]["rul"][G == h]) - lm) / ls)
            sc = np.concatenate(scores); n = len(sc)
            lam = float(np.quantile(sc, min(1.0, np.ceil((n + 1) * 0.9) / n)) / z)
        lams.append(lam)
        lm = np.log(np.median(sg, 1))[:, None]
        Scp[tem] = np.exp(lm + lam * (np.log(sg) - lm))
    r = Me.evaluate(pt, S_, R[1], R[2]); rc = Me.evaluate(pt, Scp, R[1], R[2]); rc["lambda"] = float(np.mean(lams))
    out[name] = r; out[name + " + CP"] = rc
    print(f"{name:34s} CRPS {r['CRPS']:.3f} MALE {r['MALE']:.3f} PICP {r['PICP90']:.2f} MPIW {r['MPIW90']:.2f} | CP: CRPS {rc['CRPS']:.3f} PICP {rc['PICP90']:.2f} MPIW {rc['MPIW90']:.2f} lam {rc['lambda']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(dict(pt=pt.astype(np.float32), s=S_.astype(np.float32), scp=Scp.astype(np.float32)),
                open(f"/home/claude/paper/results/ftpred_{a.pre}_k{a.k}_s{a.seed}_{name.split(',')[0].replace(' ', '').replace('.', '').replace('+', '')}_{name.split(', ')[1].replace(' ', '')}.pkl", "wb"))
json.dump(dict(pre=a.pre, seed=a.seed, ft_epochs=a.ft_epochs, ft_lr=a.ft_lr, results=out,
               c={k: R[1][k].tolist() for k in ["rul", "Kf", "tt"]}, groups=G.tolist(), idx=R[2].tolist()),
          open(f"/home/claude/paper/results/FT_{a.pre}_k{a.k}_s{a.seed}.json", "w"))
