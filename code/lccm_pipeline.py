"""
Early full pipeline: component NL and coordinate NL together.
  A1: Walsh-guided tabu search -> component NL 104 (then a short attempt at 106)
  A2: DU reduction while keeping component NL >= 104 (positions contributing to the worst DDT entries)
  A3: output-basis selection (coordinate NL, BIC-NL, SAC, BIC-SAC) + fixed-point removal by constant XOR
Control: the same pipeline started from random permutations.

Usage: python lccm_pipeline.py <n_seeds> <a2_iterations> <a106_iterations>
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


def phase2_du(s0, rng, iters):
    s = s0.copy()
    W = O.walsh(s)
    nl0 = G.key_of(W)[0]
    D = O.ddt(s)

    def dkey(D, W):
        du = D.max()
        return (-int(du), -int((D == du).sum()), G.key_of(W)[1])

    k = dkey(D, W)
    for _ in range(iters):
        du = D.max()
        rows, cols = np.nonzero(D == du)
        r = rng.randint(len(rows))
        dx, dy = rows[r] + 1, cols[r]
        xs = np.nonzero((s ^ s[IDX ^ dx]) == dy)[0]
        i = int(xs[rng.randint(len(xs))])
        j = rng.randint(SIZE - 1)
        j = j + 1 if j >= i else j
        W2 = O.swap_walsh(W, s, i, j)
        if G.key_of(W2)[0] < nl0:
            continue
        s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
        D2 = O.ddt(s2)
        k2 = dkey(D2, W2)
        if k2 >= k:
            s, W, D, k = s2, W2, D2, k2
    return s


def job(args):
    start, sid, key, a2_iters, a106_iters = args
    if start == "lccm":
        s0, _ = L.generate_sbox(P, *key)
    else:
        s0 = np.random.default_rng(5000 + sid).permutation(SIZE).astype(np.int64)
    rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    s1, k1, hit104 = G.guided_tabu(s0, rng, 20000, target=104)
    t1 = time.time() - t0
    # attempt at 106 (continue with the NL-104 S-box if it fails)
    s106, k106, hit106 = G.guided_tabu(s1, rng, a106_iters, target=106)
    t106 = time.time() - t0 - t1
    base = s106 if k106[0] >= 106 else s1
    s2 = phase2_du(base, rng, a2_iters)
    s3, masks, c = Dl.phase3(s2, rng, trials=300)
    return dict(start=start, sid=sid, key=key, s0=s0, s1=s1, s2=s2, s3=s3, masks=masks, c=c,
                hit104=hit104, k106=k106, hit106=hit106, t1=t1, t106=t106, ttot=time.time() - t0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a2 = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    a106 = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    g = np.random.default_rng(2026)
    ks = [tuple(map(float, g.uniform(0.05, 0.95, 2))) for _ in range(n)]
    jobs = [(st, k, ks[k], a2, a106) for st in ("lccm", "random") for k in range(n)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"Boru hatti: {n} tohum x (2D-LCCM, rastgele), A2 {a2} deneme, 106 denemesi {a106} iterasyon")
    res = []
    with Pool(11) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            say(f"  {r['start']:<6} t{r['sid']}: 104@{r['hit104']} it ({r['t1']:.0f}s), "
                f"106 denemesi -> {r['k106'][0]} ({r['t106']:.0f}s), toplam {r['ttot']:.0f}s")

    A = {}
    for r in res:
        for st in ("s0", "s1", "s2", "s3"):
            A[(r["start"], r["sid"], st)] = L.analyze(r[st])

    say("\n== Kontrol deneyi: 2D-LCCM vs rastgele baslangic ==")
    for st in ("lccm", "random"):
        rs = [r for r in res if r["start"] == st]
        h = [r["hit104"] for r in rs if r["hit104"] is not None]
        say(f"  {st:<6}: baslangic cNL ort {np.mean([A[(st, r['sid'], 's0')]['comp_NL'] for r in rs]):.1f}, "
            f"104'e ulasan {len(h)}/{len(rs)}, iterasyon ort {np.mean(h):.0f} (min {min(h)}, maks {max(h)}), "
            f"106'ya ulasan {sum(r['k106'][0] >= 106 for r in rs)}/{len(rs)}, "
            f"son DU ort {np.mean([A[(st, r['sid'], 's3')]['DU'] for r in rs]):.2f}, "
            f"son LP ort {np.mean([A[(st, r['sid'], 's3')]['LP'] for r in rs]):.4f}, "
            f"son koord NL ort {np.mean([A[(st, r['sid'], 's3')]['NL_avg'] for r in rs]):.2f}")

    say("\n== Asamalar (her tohum icin: A0 ham, A1 NL104, A2 DU, A3 taban+sabit) ==")
    say(L.HEADER)
    for st in ("lccm", "random"):
        for r in sorted([r for r in res if r["start"] == st], key=lambda z: z["sid"]):
            for ph, lab in (("s0", "A0"), ("s1", "A1"), ("s2", "A2"), ("s3", "A3")):
                say(L.row(f"{st} t{r['sid']} {lab}", A[(st, r["sid"], ph)]))

    def rank(r):
        a = A[(r["start"], r["sid"], "s3")]
        return (a["comp_NL"], -a["DU"], -a["LP"], a["NL_min"], a["NL_avg"], a["BIC_NL_min"], a["BIC_NL_avg"],
                -abs(a["SAC_avg"] - 0.5), -abs(a["BIC_SAC_avg"] - 0.5), -a["fixed_points"])

    lc = sorted([r for r in res if r["start"] == "lccm"], key=rank, reverse=True)
    best = lc[0]
    a = A[("lccm", best["sid"], "s3")]
    say(f"\n== Onerilen S-box: 2D-LCCM t{best['sid']} (anahtar x0={best['key'][0]:.10f}, y0={best['key'][1]:.10f}) ==")
    say(f"Cikis tabani maskeleri: {[f'{m:08b}' for m in best['masks']]}, cikis sabiti: {best['c']}")
    say(L.HEADER)
    say(L.row("AES (referans)", L.analyze(L.aes_sbox())))
    say(L.row("onerilen", a))
    say(L.fmt_sbox(best["s3"]))
    say(f"Koordinat NL: {a['NL_all']}")
    say(f"Cebirsel dereceler: {L.algebraic_degree(best['s3'])}")
    say(f"SAC min/maks: {a['SAC_min']:.4f}/{a['SAC_max']:.4f}; BIC-SAC min/maks: {a['BIC_SAC_min']:.4f}/{a['BIC_SAC_max']:.4f}")
    say("SAC matrisi:")
    for rr in a["SAC_matrix"]:
        say("  " + " ".join(f"{v:.4f}" for v in rr))
    np.save("lccm_pipeline_best.npy", best["s3"])
    with open("lccm_pipeline_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
