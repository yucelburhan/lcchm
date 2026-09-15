"""
Walsh+DDT-guided tabu search that accounts for DU during the nonlinearity climb (comparison of variants).

Usage: python lccm_du2.py <n_seeds> <iterations>
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

SIZE = 256
IDX = np.arange(SIZE)
P = O.P


def stats(W, D):
    aw = np.abs(W)
    m = aw.max()
    du = int(D.max())
    return dict(nl=int(SIZE // 2 - m // 2), cnt=int((aw == m).sum()), cnt4=int((aw == m - 4).sum()),
                du=du, ndu=int((D == du).sum()))


KEYS = {
    "V1": lambda t: (t["nl"], -t["cnt"], -t["du"], -t["ndu"], -t["cnt4"]),
    "V2": lambda t: (t["nl"], -max(t["du"], 8), -t["cnt"], -t["du"], -t["ndu"], -t["cnt4"]),
    "V3": lambda t: (t["nl"], -t["du"], -t["ndu"], -t["cnt"], -t["cnt4"]),
}


def search(s0, rng, iters, variant, nl_target=104, Kw=20, Kd=20, tenure=12):
    keyf = KEYS[variant]
    s = s0.copy(); W = O.walsh(s); D = O.ddt(s)
    t = stats(W, D)
    best = None  # among states with NL >= 104: (du, ndu)
    first104 = None
    visits = 0
    tabu_until = np.zeros(SIZE, dtype=np.int64)
    for it in range(1, iters + 1):
        sc = G.pair_scores(s, W)
        free = tabu_until <= it
        sc = np.where(G.TRIU & free[:, None] & free[None, :], sc, -1e18)
        flat = np.argpartition(sc.ravel(), -Kw)[-Kw:]
        pairs = {divmod(int(f), SIZE) for f in flat}
        rows, cols = np.nonzero(D == D.max())
        hot = set()
        for r, c in zip(rows, cols):
            hot.update(np.nonzero((s ^ s[IDX ^ (r + 1)]) == c)[0].tolist())
        hot = sorted(hot)
        full = np.maximum(sc, sc.T)
        for _ in range(Kd):
            i = hot[rng.randint(len(hot))]
            top = np.argpartition(full[i], -4)[-4:]
            j = int(top[rng.randint(4)])
            if full[i, j] > -1e17 and i != j:
                pairs.add((min(i, j), max(i, j)))
        cand = None
        for i, j in pairs:
            W2 = O.swap_walsh(W, s, i, j)
            s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
            D2 = O.ddt(s2)
            t2 = stats(W2, D2)
            k2 = keyf(t2)
            if cand is None or k2 > cand[0]:
                cand = (k2, s2, W2, D2, t2, i, j)
        _, s, W, D, t, i, j = cand
        tabu_until[i] = tabu_until[j] = it + tenure
        if t["nl"] >= nl_target:
            visits += 1
            if first104 is None:
                first104 = it
            b = (-t["du"], -t["ndu"])
            if best is None or b > best[0]:
                best = (b, s.copy(), it)
                if t["du"] <= 6:
                    break
    return best, first104, visits


def job(args):
    variant, sid, key, iters = args
    s0, _ = L.generate_sbox(P, *key)
    rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    best, first104, visits = search(s0, rng, iters, variant)
    return variant, sid, key, best, first104, visits, time.time() - t0


def main():
    n = int(sys.argv[1]); iters = int(sys.argv[2])
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else list(KEYS)
    g = np.random.default_rng(2026)
    ks = [tuple(map(float, g.uniform(0.05, 0.95, 2))) for _ in range(n)]
    jobs = [(v, k, ks[k], iters) for v in variants for k in range(n)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"Varyantlar {variants}, {n} anahtar, {iters} iterasyon")
    res = []
    with Pool(11) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            v, sid, key, best, f104, visits, dt = r
            bs = f"DU={-best[0][0]} (#{-best[0][1]}) @it{best[2]}" if best else "NL104 yok"
            say(f"  {v} t{sid}: ilk104@{f104}, 104 ziyaret={visits}, en iyi {bs}, {dt:.0f}s")
    say("\nOzet:")
    for v in variants:
        rs = [r for r in res if r[0] == v]
        dus = [-r[3][0][0] for r in rs if r[3]]
        say(f"  {v}: NL104'e ulasan {len(dus)}/{len(rs)}, DU dagilimi {dict(zip(*np.unique(dus, return_counts=True))) if dus else {}}")
    np.save("lccm_du2_sboxes.npy", np.array([r[3][1] for r in res if r[3]]))
    with open("lccm_du2_meta.txt", "w", encoding="utf-8") as f:
        for r in res:
            if r[3]:
                f.write(f"{r[0]} t{r[1]} key={r[2]} DU={-r[3][0][0]} n={-r[3][0][1]}\n")
    with open("lccm_du2_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
