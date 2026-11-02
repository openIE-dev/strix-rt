"""Cactus-style hidden-state probe router vs confidence routing (venv-kt python: numpy).
probe.py T1 [T2=t2]: logistic-regression probe on tier-1 <decide> hidden states predicting P(tier-1 wrong).
Train on dev (first 50/dataset, 200 items; L2 picked by 5-fold CV on dev), evaluate on test (800). Also a 5-fold
out-of-fold variant over the test split (more training data; reported separately). Routing: escalate the top-r fraction
by score (confidence: lowest refit max-prob; probe: highest P(wrong)); cascade J = e_s + escalated * e_l per dataset (median gross)."""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import analyze as A
t1 = sys.argv[1]; t2 = sys.argv[2] if len(sys.argv) > 2 else "t2"; D = A.D
H = {r["id"]: r for r in map(json.loads, open(f"{D}/hidden/{t1}.jsonl"))}
r1 = {}; [r1.update(A.load(f"{D}/acc/{t1}_{d}.jsonl")) for d in A.DS]; r1 = A.apply_T(r1, A.refit(r1))
r2 = {}; [r2.update(A.load(f"{D}/acc/{t2}_{d}.jsonl")) for d in A.DS]
E = A.energy(); es = {d: A.med(E[(t1, d)]["gross"]) for d in A.DS}; el = {d: A.med(E[(t2, d)]["gross"]) for d in A.DS}
ids = [i for i in H if i in r1 and i in r2]; dev = [i for i in ids if A.split(i) == "dev"]; te = [i for i in ids if A.split(i) == "test"]
X = lambda L: np.array([H[i]["h"] + [r1[i]["conf"], np.log(max(1 - r1[i]["conf"], 1e-6))] for i in L], dtype=np.float64)
y = lambda L: np.array([float(r1[i]["pred"] != r1[i]["gold"]) for i in L])
def fit(Xa, ya, lam):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(400):  # full-batch gradient descent with L2
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - ya; w -= 0.1 * (Z.T @ g / len(ya) + lam * w); b -= 0.1 * g.mean()
    return lambda Xb: 1 / (1 + np.exp(-(((Xb - mu) / sd) @ w + b)))
def auc(s, yy):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); P = yy.sum(); N = len(yy) - P
    return (r[yy == 1].sum() - P * (P + 1) / 2) / (P * N)
def cvlam(Xa, ya, k=5):
    best = None; f = np.arange(len(ya)) % k
    for lam in [1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]:
        s = np.zeros(len(ya))
        for j in range(k): m = f == j; s[m] = fit(Xa[~m], ya[~m], lam)(Xa[m])
        a = auc(s, ya); best = max(best or (a, lam), (a, lam))
    return best[1]
Xd, yd, Xt, yt = X(dev), y(dev), X(te), y(te)
lam = cvlam(Xd, yd); s_dev = fit(Xd, yd, lam)(Xt)
f = np.arange(len(te)) % 5; s_oof = np.zeros(len(te)); lam2 = cvlam(Xt, yt)
for j in range(5): m = f == j; s_oof[m] = fit(Xt[~m], yt[~m], lam2)(Xt[m])
s_conf = np.array([1 - r1[i]["conf"] for i in te])
s_orc = yt + 1e-3 * s_conf
ok1 = np.array([r1[i]["pred"] == r1[i]["gold"] for i in te]); ok2 = np.array([r2[i]["pred"] == r2[i]["gold"] for i in te])
eS = np.array([es[r1[i]["ds"]] for i in te]); eL = np.array([el[r1[i]["ds"]] for i in te]); n = len(te)
jl = eL.mean(); al = ok2.mean()
print(f"## Probe router {t1} -> {t2} (test n={n}; tier-1 wrong rate {yt.mean():.3f}; dev L2 {lam}, oof L2 {lam2})\n")
print(f"AUROC for predicting tier-1 wrong: confidence {auc(s_conf, yt):.3f}, probe(dev-trained) {auc(s_dev, yt):.3f}, probe(5-fold oof on test) {auc(s_oof, yt):.3f}\n")
print(f"Always-large: acc {al:.3f}, {jl:.2f} J/dec, {jl/al:.2f} J/correct. Always-small: acc {ok1.mean():.3f}, {eS.mean():.2f} J/dec\n")
print("| escalate | conf acc | conf J/correct | probe(dev) acc | probe(dev) J/correct | probe(oof) acc | probe(oof) J/correct | oracle acc | oracle J/correct |\n|---|---|---|---|---|---|---|---|---|")
res = {}
for r in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    k = int(round(r * n)); row = [f"{r:.2f}"]
    for name, s in (("conf", s_conf), ("dev", s_dev), ("oof", s_oof), ("orc", s_orc)):
        esc = np.zeros(n, bool); esc[np.argsort(-s)[:k]] = True
        acc = np.where(esc, ok2, ok1).mean(); J = (eS + esc * eL).mean(); row += [f"{acc:.3f}", f"{J/acc:.2f}"]; res[(name, r)] = (acc, J)
    print("| " + " | ".join(row) + " |")
best = max(r for (nm, r) in res if nm == "conf")
print(f"\nJ/decision at each escalation rate is the same for every router (same count escalated, small per-dataset differences only).")
