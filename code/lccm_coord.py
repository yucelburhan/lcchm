"""
Full design search: maximise the attainable coordinate nonlinearity while keeping component NL 104 and DU 8.

Attainable coordinate-NL sum ("basis sum"): sort the 255 components by nonlinearity and greedily keep the
first 8 that are linearly independent over GF(2) (greedy is optimal on a linear matroid).
The basis sum is part of the lexicographic search key; the search continues after reaching 104 and keeps,
among the visited states with NL 104 and DU <= 8, the one with the largest basis sum.

Usage: python lccm_coord.py <n_seeds> <iterations> [C1,C2]
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
MASKS = np.arange(1, SIZE)


def basis_sum(W):
    comp_nl = SIZE // 2 - (np.abs(W).max(axis=1) // 2).astype(np.int64)
    order = np.argsort(-comp_nl, kind="stable")
    basis, total, n = [], 0, 0
    for idx in order:
        v = int(MASKS[idx])
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v); total += int(comp_nl[idx]); n += 1
            if n == 8:
                break
    return total


def stats(W, D):
    aw = np.abs(W)
    m = aw.max()
    du = int(D.max())
    return dict(nl=int(SIZE // 2 - m // 2), cnt=int((aw == m).sum()), cnt4=int((aw == m - 4).sum()),
                du=du, ndu=int((D == du).sum()), bsum=basis_sum(W))


KEYS = {
    "C1": lambda t: (t["nl"], -max(t["du"], 8), -t["cnt"], t["bsum"], -t["du"], -t["ndu"], -t["cnt4"]),
    "C2": lambda t: (t["nl"], -max(t["du"], 8), t["bsum"], -t["cnt"], -t["du"], -t["ndu"], -t["cnt4"]),
}


def search(s0, rng, iters, variant, nl_target=104, Kw=20, Kd=20, tenure=12):
    keyf = KEYS[variant]
    s = s0.copy(); W = O.walsh(s); D = O.ddt(s)
    best, first104, visits = None, None, 0
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
            b = (-max(t["du"], 8), t["bsum"], -t["du"], -t["ndu"])
            if best is None or b > best[0]:
                best = (b, s.copy(), it)
    return best, first104, visits


def job(args):
    variant, sid, key, iters = args
    s0, _ = L.generate_sbox(P, *key)
    rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    best, first104, visits = search(s0, rng, iters, variant)
    fin = None
    if best:
        t, masks, c = Dl.phase3(best[1], O.ChaosRNG(0.37, 0.59), trials=400)
        fin = (t, masks, c)
    return variant, sid, key, best, first104, visits, fin, time.time() - t0


def main():
    n = int(sys.argv[1]); iters = int(sys.argv[2])
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else list(KEYS)
    g = np.random.default_rng(2026)
    ks = [tuple(map(float, g.uniform(0.05, 0.95, 2))) for _ in range(n)]
    jobs = [(v, k, ks[k], iters) for v in variants for k in range(n)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"Koordinat NL adimi: varyantlar {variants}, {n} anahtar, {iters} iterasyon")
    res = []
    with Pool(11) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            v, sid, key, best, f104, visits, fin, dt = r
            bs = (f"DU={-best[0][2]} taban toplami={best[0][1]} (koord ort {best[0][1] / 8:.2f}) @it{best[2]}"
                  if best else "NL104 yok")
            say(f"  {v} t{sid}: ilk104@{f104}, 104 ziyaret={visits}, en iyi {bs}, {dt:.0f}s")

    say("\n" + L.HEADER)
    finals = []
    for r in sorted(res, key=lambda z: (z[0], z[1])):
        if r[6]:
            a = L.analyze(r[6][0])
            finals.append((a, r))
            say(L.row(f"{r[0]} t{r[1]}", a))
    say("\nOzet:")
    for v in variants:
        fs = [a for a, r in finals if r[0] == v]
        if fs:
            say(f"  {v}: NL104 {len(fs)}/{n}, DU {dict(zip(*np.unique([a['DU'] for a in fs], return_counts=True)))}, "
                f"koord NL ort {np.mean([a['NL_avg'] for a in fs]):.2f} (maks {max(a['NL_avg'] for a in fs):.2f}), "
                f"koord NL min ort {np.mean([a['NL_min'] for a in fs]):.1f}")

    def rank(z):
        a = z[0]
        return (a["comp_NL"], -a["DU"], -a["LP"], a["NL_min"], a["NL_avg"], a["BIC_NL_min"], a["BIC_NL_avg"],
                -abs(a["SAC_avg"] - 0.5), -abs(a["BIC_SAC_avg"] - 0.5), -a["fixed_points"])

    a, r = max(finals, key=rank)
    t, masks, c = r[6]
    D = O.ddt(t)
    say(f"\n== Onerilen S-box: {r[0]} t{r[1]} (anahtar x0={r[2][0]:.10f}, y0={r[2][1]:.10f}) ==")
    say(f"Cikis tabani maskeleri: {[f'{x:08b}' for x in masks]}, cikis sabiti: {c}")
    say(L.HEADER)
    say(L.row("AES (referans)", L.analyze(L.aes_sbox())))
    say(L.row("onerilen", a))
    say(L.fmt_sbox(t))
    say(f"Koordinat NL: {a['NL_all']}")
    say(f"DDT'de 8 degerli girdi sayisi: {int((D == 8).sum())}")
    say(f"Cebirsel dereceler: {L.algebraic_degree(t)}")
    say(f"SAC min/maks: {a['SAC_min']:.4f}/{a['SAC_max']:.4f}; BIC-SAC min/maks: {a['BIC_SAC_min']:.4f}/{a['BIC_SAC_max']:.4f}")
    say("SAC matrisi:")
    for rr in a["SAC_matrix"]:
        say("  " + " ".join(f"{v:.4f}" for v in rr))
    np.save("lccm_coord_best.npy", t)
    np.save("lccm_coord_all.npy", np.array([z[1][6][0] for z in finals]))
    with open("lccm_coord_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
