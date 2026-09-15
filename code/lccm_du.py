"""
DU reduction (target 6) while keeping component NL >= 104: DDT-guided tabu search (early experiment).

Each iteration:
  - positions x that contribute to a maximal DDT entry (dx,dy) are the "hot" positions;
    swapping a hot position i with some j directly breaks that solution;
  - N_CAND candidates are drawn (i hot, j arbitrary; choices from the 2D-LCChM stream);
  - candidates that drop the component NL below 104 are rejected via the incremental Walsh update,
    the DDT is recomputed for the rest;
  - the best candidate is accepted (even if worse) under the tabu rule; the best state is kept.
Then: output-basis selection + constant XOR (lccm_dual.phase3).

Usage: python lccm_du.py <n_seeds> <iterations>
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import numpy as np
from multiprocessing import Pool

import lccm_sbox as L
import lccm_optimizers as O
import lccm_guided as G
import lccm_dual as Dl

SIZE = 256
IDX = np.arange(SIZE)
P = O.P
N_CAND = 96
TENURE = 10


def dkey(D, W):
    du = int(D.max())
    aw = np.abs(W)
    return (-du, -int((D == du).sum()), -int((D == du - 2).sum()), -int((aw == aw.max()).sum()))


def feasible_pairs(s, W, nl_min=104):
    """Swaps that do not push any |W| above (128-nl_min)*2: F[i,j] True means the component NL is preserved."""
    thr = (SIZE // 2 - nl_min) * 2
    rows, cols = np.nonzero(np.abs(W) == thr)
    if len(rows) == 0:
        return np.ones((SIZE, SIZE), dtype=bool)
    chi = O.CHI[rows][:, s]
    h = L.HF[cols]
    C = np.sign(W[rows, cols])[:, None] * chi * h
    Np = ((C < 0) & (chi > 0)).astype(np.float64)
    Nm = ((C < 0) & (chi < 0)).astype(np.float64)
    inc = Np.T @ Nm
    inc = inc + inc.T
    F = inc == 0
    np.fill_diagonal(F, False)
    return F


def du_tabu(s0, rng, iters, nl_min=104, target_du=6, log=None):
    s = s0.copy()
    W = O.walsh(s)
    D = O.ddt(s)
    assert G.key_of(W)[0] >= nl_min
    k = dkey(D, W)
    best_k, best_s = k, s.copy()
    tabu_until = np.zeros(SIZE, dtype=np.int64)
    hit = None
    for it in range(1, iters + 1):
        du = D.max()
        rows, cols = np.nonzero(D == du)
        hot = np.zeros(SIZE, dtype=np.int64)
        for r, c in zip(rows, cols):
            hot[np.nonzero((s ^ s[IDX ^ (r + 1)]) == c)[0]] += 1
        F = feasible_pairs(s, W, nl_min)
        free = tabu_until <= it
        F &= free[:, None] & free[None, :]
        pi, pj = np.nonzero(F & (hot > 0)[:, None])
        cand = None
        n_pairs = len(pi)
        picks = {rng.randint(n_pairs) for _ in range(N_CAND)} if n_pairs else set()
        for p in picks:
            i, j = int(pi[p]), int(pj[p])
            W2 = O.swap_walsh(W, s, i, j)
            s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
            D2 = O.ddt(s2)
            k2 = dkey(D2, W2)
            if cand is None or k2 > cand[0]:
                cand = (k2, i, j, s2, W2, D2)
        if cand is None:
            if log and it % log == 0:
                print(f"    it {it}: uygun aday yok", flush=True)
            continue
        k, i, j, s, W, D = cand
        assert G.key_of(W)[0] >= nl_min
        tabu_until[i] = tabu_until[j] = it + TENURE
        if k > best_k:
            best_k, best_s = k, s.copy()
            if -k[0] <= target_du and hit is None:
                hit = it
                break
        if log and it % log == 0:
            print(f"    it {it}: simdiki DU={-k[0]} (#{-k[1]})  en iyi DU={-best_k[0]} (#{-best_k[1]})", flush=True)
    return best_s, best_k, hit


def joint_tabu(s0, rng, iters, nl_target=104, target_du=6, Kw=24, Kd=24, tenure=12, log=None):
    """Tabu search that wanders around NL 104: may drop to 102, reduces DU while at 104."""
    s = s0.copy()
    W = O.walsh(s)
    D = O.ddt(s)
    TR = G.TRIU

    def ckey(W, D):
        aw = np.abs(W)
        m = aw.max()
        nl = int(SIZE // 2 - m // 2)
        du = int(D.max())
        at = nl >= nl_target
        return (min(nl, nl_target), 0 if at else -int((aw == m).sum()), -du, -int((D == du).sum()),
                -int((D == du - 2).sum()), -int((aw == m).sum()))

    def bkey(W, D):
        du = int(D.max())
        return (-du, -int((D == du).sum()), -int((D == du - 2).sum()))

    best_s, best_k = None, None
    if G.key_of(W)[0] >= nl_target:
        best_s, best_k = s.copy(), bkey(W, D)
    tabu_until = np.zeros(SIZE, dtype=np.int64)
    hit = None
    visits = 0
    for it in range(1, iters + 1):
        sc = G.pair_scores(s, W)
        free = tabu_until <= it
        sc = np.where(TR & free[:, None] & free[None, :], sc, -1e18)
        # Walsh candidates
        flat = np.argpartition(sc.ravel(), -Kw)[-Kw:]
        pairs = {divmod(int(f), SIZE) for f in flat}
        # DDT candidates: hot position i, best Walsh-scored j in that row
        du = D.max()
        rows, cols = np.nonzero(D == du)
        hot = set()
        for r, c in zip(rows, cols):
            hot.update(np.nonzero((s ^ s[IDX ^ (r + 1)]) == c)[0].tolist())
        hot = sorted(hot)
        full = np.maximum(sc, sc.T)
        for _ in range(Kd):
            i = hot[rng.randint(len(hot))]
            row = full[i]
            top = np.argpartition(row, -4)[-4:]
            j = int(top[rng.randint(4)])
            if row[j] > -1e17 and i != j:
                pairs.add((min(i, j), max(i, j)))
        cand = None
        for i, j in pairs:
            W2 = O.swap_walsh(W, s, i, j)
            s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
            D2 = O.ddt(s2)
            k2 = ckey(W2, D2)
            if cand is None or k2 > cand[0]:
                cand = (k2, i, j, s2, W2, D2)
        k2, i, j, s, W, D = cand
        tabu_until[i] = tabu_until[j] = it + tenure
        if G.key_of(W)[0] >= nl_target:
            visits += 1
            b = bkey(W, D)
            if best_k is None or b > best_k:
                best_k, best_s = b, s.copy()
                if -b[0] <= target_du and hit is None:
                    hit = it
                    break
        if log and it % log == 0:
            print(f"    it {it}: NL={G.key_of(W)[0]} DU={int(D.max())}  104 ziyaret={visits}  en iyi={best_k}", flush=True)
    return best_s, best_k, hit, visits


def job(args):
    sid, key, iters = args
    s0, _ = L.generate_sbox(P, *key)
    rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    s1, _, hit104 = G.guided_tabu(s0, rng, 20000, target=104)
    t1 = time.time() - t0
    s2, k2, hit6 = du_tabu(s1, rng, iters)
    t2 = time.time() - t0 - t1
    s3, masks, c = Dl.phase3(s2, rng, trials=300)
    return dict(sid=sid, key=key, s1=s1, s2=s2, s3=s3, masks=masks, c=c, hit104=hit104, hit6=hit6,
                du_final=-k2[0], t1=t1, t2=t2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    g = np.random.default_rng(2026)  # same keys as lccm_pipeline
    ks = [tuple(map(float, g.uniform(0.05, 0.95, 2))) for _ in range(n)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"DU azaltma: {n} 2D-LCCM anahtari, en fazla {iters} iterasyon, {N_CAND} aday/iterasyon, hedef DU 6")
    res = []
    with Pool(min(11, n)) as pool:
        for r in pool.imap_unordered(job, [(k, ks[k], iters) for k in range(n)]):
            res.append(r)
            say(f"  t{r['sid']}: NL104@{r['hit104']} ({r['t1']:.0f}s)  ->  DU={r['du_final']}"
                f"{' (6@' + str(r['hit6']) + ' it)' if r['hit6'] else ''} ({r['t2']:.0f}s)")
    res.sort(key=lambda r: r["sid"])
    say("\n" + L.HEADER)
    A = {}
    for r in res:
        for ph, lab in (("s1", "NL104"), ("s2", "DU"), ("s3", "son")):
            A[(r["sid"], ph)] = L.analyze(r[ph])
            say(L.row(f"t{r['sid']} {lab}", A[(r["sid"], ph)]))
    dus = [A[(r["sid"], "s3")]["DU"] for r in res]
    say(f"\nDU dagilimi (son): {dict(zip(*np.unique(dus, return_counts=True)))}; "
        f"once: {dict(zip(*np.unique([A[(r['sid'], 's1')]['DU'] for r in res], return_counts=True)))}")

    def rank(r):
        a = A[(r["sid"], "s3")]
        return (a["comp_NL"], -a["DU"], -a["LP"], a["NL_min"], a["NL_avg"], a["BIC_NL_min"], a["BIC_NL_avg"],
                -abs(a["SAC_avg"] - 0.5), -abs(a["BIC_SAC_avg"] - 0.5), -a["fixed_points"])

    best = max(res, key=rank)
    a = A[(best["sid"], "s3")]
    say(f"\n== Onerilen S-box: t{best['sid']} (anahtar x0={best['key'][0]:.10f}, y0={best['key'][1]:.10f}) ==")
    say(f"Cikis tabani maskeleri: {[f'{m:08b}' for m in best['masks']]}, cikis sabiti: {best['c']}")
    say(L.HEADER)
    say(L.row("AES (referans)", L.analyze(L.aes_sbox())))
    say(L.row("onerilen", a))
    say(L.fmt_sbox(best["s3"]))
    say(f"Koordinat NL: {a['NL_all']}")
    say(f"Cebirsel dereceler: {L.algebraic_degree(best['s3'])}")
    say(f"SAC min/maks: {a['SAC_min']:.4f}/{a['SAC_max']:.4f}; BIC-SAC min/maks: {a['BIC_SAC_min']:.4f}/{a['BIC_SAC_max']:.4f}")
    np.save("lccm_du_best.npy", best["s3"])
    np.save("lccm_du_all_s2.npy", np.array([r["s2"] for r in res]))
    with open("lccm_du_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
