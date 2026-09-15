"""
Attempt to reach DU 6 together with component NL 104.

Variants (lexicographic key, larger is better):
  D6a: (-max(DU,6), NL, -#max|W|, -#DU, -#(|W|=max-4))          DU<=6 first, then NL
  D6b: (min(NL,102), -max(DU,6), NL, -#max|W|, -#DU, -#(max-4)) NL 102 first, then DU<=6, then NL 104
Candidate swaps: the Kw best Walsh-scored swaps + Kd swaps involving positions of the worst DDT entries.
The best (NL, -DU) state reached is recorded.

Usage: python lccm_du6.py <n_seeds> <iterations> [D6a,D6b]
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


def stats(W, D):
    aw = np.abs(W)
    m = aw.max()
    du = int(D.max())
    return dict(nl=int(SIZE // 2 - m // 2), cnt=int((aw == m).sum()), cnt4=int((aw == m - 4).sum()),
                du=du, ndu=int((D == du).sum()))


KEYS = {
    "D6a": lambda t: (-max(t["du"], 6), t["nl"], -t["cnt"], -t["ndu"], -t["cnt4"]),
    "D6b": lambda t: (min(t["nl"], 102), -max(t["du"], 6), t["nl"], -t["cnt"], -t["ndu"], -t["cnt4"]),
}


def search(s0, rng, iters, variant, Kw=20, Kd=28, tenure=12):
    keyf = KEYS[variant]
    s = s0.copy(); W = O.walsh(s); D = O.ddt(s)
    t = stats(W, D)
    best = ((int(t["du"] <= 6), t["nl"], -t["du"], -t["ndu"]), s.copy(), 0)
    first_du6 = None
    trace = []
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
            if rng.rand() < 0.5:
                top = np.argpartition(full[i], -4)[-4:]
                j = int(top[rng.randint(4)])
            else:
                j = rng.randint(SIZE)
            if i != j and free[i] and free[j]:
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
        if t["du"] <= 6 and first_du6 is None:
            first_du6 = it
        b = (int(t["du"] <= 6), t["nl"], -t["du"], -t["ndu"])  # DU <= 6 first, then NL
        if b > best[0]:
            best = (b, s.copy(), it)
        if it % 250 == 0:
            trace.append((it, t["nl"], t["du"]))
        if t["du"] <= 6 and t["nl"] >= 104:
            break
    return best, first_du6, trace


def job(args):
    variant, sid, key, iters = args
    s0, _ = L.generate_sbox(P, *key)
    rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    best, first_du6, trace = search(s0, rng, iters, variant)
    return variant, sid, key, best, first_du6, trace, time.time() - t0


def main():
    n = int(sys.argv[1]); iters = int(sys.argv[2])
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else list(KEYS)
    g = np.random.default_rng(2026)
    ks = [tuple(map(float, g.uniform(0.05, 0.95, 2))) for _ in range(n)]
    jobs = [(v, k, ks[k], iters) for v in variants for k in range(n)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"DU 6 denemesi: varyantlar {variants}, {n} anahtar, {iters} iterasyon")
    res = []
    with Pool(11) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            v, sid, key, best, fdu6, trace, dt = r
            b = best[0]
            say(f"  {v} t{sid}: ilk DU6@{fdu6}, en iyi: DU<=6={bool(b[0])} NL={b[1]} DU={-b[2]} (#{-b[3]}) @it{best[2]}, "
                f"iz {' '.join(f'{i}:{nl}/{du}' for i, nl, du in trace[::2])}  ({dt:.0f}s)")
    say("\nOzet (DU<=6 olan durumlarda ulasilan en yuksek bilesen NL):")
    for v in variants:
        rs = [r for r in res if r[0] == v]
        say(f"  {v}: " + ", ".join(f"t{r[1]}: " + (f"NL {r[3][0][1]} @DU{-r[3][0][2]}" if r[3][0][0] else
                                                  f"DU6 yok (en iyi NL {r[3][0][1]} DU {-r[3][0][2]})") for r in sorted(rs, key=lambda z: z[1])))
    good = [r for r in res if r[3][0][0]]
    if good:
        r = max(good, key=lambda z: z[3][0])
        s = r[3][1]
        t, masks, c = Dl.phase3(s, O.ChaosRNG(0.37, 0.59), trials=400)
        a = L.analyze(t)
        say(f"\n== DU<=6 olan en iyi S-box: {r[0]} t{r[1]} (anahtar x0={r[2][0]!r}, y0={r[2][1]!r}) ==")
        say(L.HEADER)
        say(L.row("DU6 en iyi", a))
        np.save("lccm_du6_best.npy", t)
    with open("lccm_du6_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
