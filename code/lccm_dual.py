"""
Component NL and coordinate NL together: three-phase 2D-LCChM S-box design (early version).

Phase 1: simulated annealing raises the component NL (min over 255 components) towards 104.
Phase 2: improve DU and LP without lowering the component NL (hill climbing).
Phase 3: output-basis selection. Component NL, LP and DU are INVARIANT under an invertible linear map of
         the output; the coordinate NLs are not. The 8 linearly independent components with the highest NL
         become the new output bits; coordinate NL, BIC-NL, SAC and BIC-SAC are optimised accordingly.
         Fixed points are then removed by XOR-ing a constant to the output (all metrics preserved).

Usage:
  python lccm_dual.py tune  <budget> <n_seeds>          # cost-function tuning
  python lccm_dual.py full  <budget> <n_seeds> <cost>   # full pipeline + random-start control
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import time
import numpy as np
from multiprocessing import Pool

import lccm_sbox as L
import lccm_optimizers as O

P = O.P
SIZE = 256
IDX = np.arange(SIZE)
WV = np.arange(SIZE + 1, dtype=np.float64)

COSTS = {
    "whs_x0_r3":   WV ** 3,
    "whs_x32_r3":  np.abs(WV - 32) ** 3,
    "whs_x16_r4":  np.abs(WV - 16) ** 4,
    "relu32_r3":   np.maximum(0, WV - 32) ** 3,
    "relu36_r4":   np.maximum(0, WV - 36) ** 4,
    "relu40_r6":   np.maximum(0, WV - 40) ** 6,
    "exp4_w":      4.0 ** ((WV - 32) / 4.0),
    "exp8_w44":    np.where(WV >= 44, 8.0 ** ((WV - 44) / 4.0 + 1), 0.0),
}


def wcost(table, W):
    return float(table[np.abs(W).astype(np.int64)].sum())


def cnl_of(W):
    aw = np.abs(W)
    wm = aw.max()
    return int(SIZE // 2 - wm // 2), int((aw == wm).sum())


# ----------------------------------------------------------------------------
# Phase 1: SA on the Walsh cost only (no DDT -> fast). Stops when the target is reached.
# ----------------------------------------------------------------------------
def phase1(s0, rng, budget, table, target=104):
    s = s0.copy(); W = O.walsh(s); c = wcost(table, W)
    best = (cnl_of(W)[0], -cnl_of(W)[1]); best_s = s.copy()
    ev = 0
    ups = []
    for _ in range(200):
        ev += 1
        i, j = rng.pair()
        dc = wcost(table, O.swap_walsh(W, s, i, j)) - c
        if dc > 0:
            ups.append(dc)
    T = (np.mean(ups) if ups else 1.0) / math.log(2.0)
    alpha = (1e-4) ** (1.0 / max(1, budget - ev))
    hit = None
    while ev < budget:
        ev += 1
        i, j = rng.pair()
        W2 = O.swap_walsh(W, s, i, j)
        c2 = wcost(table, W2)
        dc = c2 - c
        if dc <= 0 or rng.rand() < math.exp(-dc / T):
            s[i], s[j] = s[j], s[i]
            W, c = W2, c2
            nl, cnt = cnl_of(W)
            if (nl, -cnt) > best:
                best, best_s = (nl, -cnt), s.copy()
                if nl >= target and hit is None:
                    hit = ev
                    break
        T *= alpha
    return best_s, best[0], -best[1], hit, ev


# ----------------------------------------------------------------------------
# Phase 2: improve DU/LP while keeping component NL >= target
# ----------------------------------------------------------------------------
def phase2(s0, rng, budget):
    s = s0.copy(); W = O.walsh(s); D = O.ddt(s)
    nl0 = cnl_of(W)[0]

    def key(W, D):
        nl, cnt = cnl_of(W)
        du = D.max()
        return (nl, -int(du), -int((D == du).sum()), -cnt)

    k = key(W, D)
    for _ in range(budget):
        i, j = rng.pair()
        W2 = O.swap_walsh(W, s, i, j)
        if cnl_of(W2)[0] < nl0:
            continue
        s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
        D2 = O.ddt(s2)
        k2 = key(W2, D2)
        if k2 >= k:
            s, W, D, k = s2, W2, D2, k2
    return s


# ----------------------------------------------------------------------------
# Phase 3: output-basis selection + fixed-point removal
# ----------------------------------------------------------------------------
def gf2_insert(basis, v):
    for b in basis:
        v = min(v, v ^ b)
    return v


def apply_basis(s, masks):
    out = np.zeros(SIZE, dtype=np.int64)
    for i, b in enumerate(masks):
        out |= (L.POP[s & b] % 2) << i
    return out


def phase3(s, rng, trials=400):
    W = O.walsh(s)
    comp_nl = SIZE // 2 - (np.abs(W).max(axis=1) // 2).astype(np.int64)  # (255,) mask = idx+1
    masks_all = np.arange(1, SIZE)
    best = None
    for t in range(trials):
        # sort by NL; break ties with the chaotic stream
        noise = np.array([rng.rand() for _ in range(SIZE - 1)])
        order = np.lexsort((noise, -comp_nl))
        basis, chosen = [], []
        for idx in order:
            m = int(masks_all[idx])
            r = gf2_insert(basis, m)
            if r:
                basis.append(r); chosen.append(m)
                if len(chosen) == 8:
                    break
        # the order of the output bits is also randomised by the chaotic stream
        T = apply_basis(s, chosen)
        a = L.analyze(T)
        k = (a["NL_min"], a["NL_avg"], a["BIC_NL_min"], a["BIC_NL_avg"],
             -abs(a["SAC_avg"] - 0.5), -abs(a["BIC_SAC_avg"] - 0.5))
        if best is None or k > best[0]:
            best = (k, T, chosen)
    T = best[1]
    # fixed points: XOR a constant to the output (all metrics preserved); fewest fixed points, smallest c on ties
    fps = [(int(((T ^ c) == IDX).sum()), c) for c in range(SIZE)]
    fp, c = min(fps)
    return T ^ c, best[2], c


# ----------------------------------------------------------------------------
def job_tune(args):
    cost_name, seed_id, key, budget, start = args
    if start == "lccm":
        s0, _ = L.generate_sbox(P, *key)
        rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    else:
        s0 = np.random.default_rng(1000 + seed_id).permutation(SIZE).astype(np.int64)
        rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    s, nl, cnt, hit, ev = phase1(s0, rng, budget, COSTS[cost_name])
    return cost_name, seed_id, start, nl, cnt, hit, ev, time.time() - t0, s


def job_full(args):
    cost_name, seed_id, key, budget, start = args
    r = job_tune(args)
    s1 = r[8]
    rng = O.ChaosRNG(0.3 * key[0] + 0.41, 0.3 * key[1] + 0.29)
    t0 = time.time()
    s2 = phase2(s1, rng, 20000)
    s3, masks, c = phase3(s2, rng)
    return r[:8] + (s1, s2, s3, masks, c, time.time() - t0)


def keys(n, seed=11):
    g = np.random.default_rng(seed)
    return [tuple(map(float, g.uniform(0.05, 0.95, 2))) for _ in range(n)]


def main():
    mode = sys.argv[1]
    budget = int(sys.argv[2])
    n = int(sys.argv[3])
    ks = keys(n)
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    if mode == "tune":
        jobs = [(c, k, ks[k], budget, "lccm") for c in COSTS for k in range(n)]
        say(f"AYAR: {len(COSTS)} maliyet x {n} tohum, butce {budget}, hedef bilesen NL 104")
        res = []
        with Pool(11) as pool:
            for r in pool.imap_unordered(job_tune, jobs):
                res.append(r)
                say(f"  {r[0]:<12} t{r[1]}  cNL={r[3]} (#max|W|={r[4]})  hedef@{r[5]}  deg={r[6]}  {r[7]:.0f}s")
        say("\nOzet:")
        for c in COSTS:
            rs = [r for r in res if r[0] == c]
            hits = [r[5] for r in rs if r[5] is not None]
            say(f"  {c:<12} cNL={[r[3] for r in rs]}  104'e ulasan {len(hits)}/{len(rs)}  "
                f"ort. degerlendirme={np.mean(hits) if hits else float('nan'):.0f}")
        fn = "lccm_dual_tune.txt"
    else:
        cost = sys.argv[4]
        jobs = [(cost, k, ks[k], budget, st) for st in ("lccm", "random") for k in range(n)]
        say(f"TAM: maliyet {cost}, {n} tohum x (2D-LCCM / rastgele baslangic), asama-1 butce {budget}")
        res = []
        with Pool(11) as pool:
            for r in pool.imap_unordered(job_full, jobs):
                res.append(r)
                say(f"  {r[2]:<6} t{r[1]}  cNL={r[3]}  hedef@{r[5]}  asama1 {r[7]:.0f}s  asama2+3 {r[14]:.0f}s")
        say("\n== Kontrol deneyi: 104'e ulasma ==")
        for st in ("lccm", "random"):
            rs = [r for r in res if r[2] == st]
            hits = [r[5] for r in rs if r[5] is not None]
            say(f"  {st:<6}: ulasan {len(hits)}/{len(rs)}, ort. degerlendirme "
                f"{np.mean(hits) if hits else float('nan'):.0f}, ham baslangic ort. cNL "
                f"{np.mean([L.analyze(L.generate_sbox(P, *ks[r[1]])[0])['comp_NL'] if st == 'lccm' else L.analyze(np.random.default_rng(1000 + r[1]).permutation(SIZE).astype(np.int64))['comp_NL'] for r in rs]):.1f}")
        say("\n" + L.HEADER)
        for st in ("lccm", "random"):
            for r in sorted([r for r in res if r[2] == st], key=lambda z: z[1]):
                say(L.row(f"{st} t{r[1]} A1", L.analyze(r[8])))
                say(L.row(f"{st} t{r[1]} A2", L.analyze(r[9])))
                say(L.row(f"{st} t{r[1]} A3", L.analyze(r[10])))
        lc = [r for r in res if r[2] == "lccm"]
        lc.sort(key=lambda r: (L.analyze(r[10])["comp_NL"], -L.analyze(r[10])["DU"], L.analyze(r[10])["NL_avg"],
                               L.analyze(r[10])["BIC_NL_avg"], -abs(L.analyze(r[10])["SAC_avg"] - 0.5)), reverse=True)
        r = lc[0]
        a = L.analyze(r[10])
        say(f"\n== Onerilen S-box (2D-LCCM t{r[1]}, anahtar x0={ks[r[1]][0]:.10f}, y0={ks[r[1]][1]:.10f}) ==")
        say(f"Cikis tabani maskeleri: {[f'{m:08b}' for m in r[11]]}, cikis sabiti: {r[12]}")
        say(L.HEADER)
        say(L.row("onerilen", a))
        say(L.fmt_sbox(r[10]))
        say(f"Koordinat NL: {a['NL_all']}")
        say(f"Cebirsel dereceler: {L.algebraic_degree(r[10])}")
        say(f"BIC-SAC min/maks: {a['BIC_SAC_min']:.4f} / {a['BIC_SAC_max']:.4f}")
        say("SAC matrisi:")
        for rr in a["SAC_matrix"]:
            say("  " + " ".join(f"{v:.4f}" for v in rr))
        np.save("lccm_dual_best.npy", r[10])
        fn = "lccm_dual_full.txt"
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
