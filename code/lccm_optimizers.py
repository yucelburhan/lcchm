"""
Comparison of classical optimisers on 2D-LCChM raw S-boxes.
All random decisions (swap positions, acceptance, selection, crossover) come from the 2D-LCChM stream.

Methods:
  HC-lex   : hill climbing, lexicographic objective (component NL, -DU, ...)
  SA-WHS   : simulated annealing, Walsh-Hadamard spectrum cost sum|W|^3 (Clark-type)
  SA-EXP   : simulated annealing, exponential Walsh cost + DDT penalty
  TABU-EXP : tabu search (K candidate swaps per step), same exponential cost
  GA-EXP   : genetic algorithm (tournament, cycle crossover, swap mutation, elitism)
Every method uses the same number of objective evaluations for a fair comparison.
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

P = {"mu": 3.99, "rho": 2.59, "k": 5.3, "lam": 0.7}
SIZE = 256
IDX = np.arange(SIZE)
HF = L.HF
CHI = (1.0 - 2.0 * L.COMP_MASK)  # (255,256): (-1)^(b.y)
XD = IDX[1:, None] ^ IDX[None, :]
ROWOFF = (np.arange(SIZE - 1) * SIZE)[:, None]

W_ABS_IDX = np.arange(SIZE + 1)
TW_EXP = 4.0 ** ((W_ABS_IDX - 32) / 4.0)
TW_WHS = (W_ABS_IDX.astype(np.float64) ** 3) / 1e3
TD_EXP = np.where(W_ABS_IDX >= 6, 4.0 * 4.0 ** ((W_ABS_IDX - 4) / 2.0), 0.0)


# ----------------------------------------------------------------------------
class ChaosRNG:
    def __init__(self, x0, y0, p=P, discard=1000):
        self.x, self.y, self.n, self.p = x0, y0, 0, p
        for _ in range(discard):
            self._step()

    def _step(self):
        self.x, self.y = L.lccm_step(self.x, self.y, self.n, self.p)
        self.n += 1

    def rand(self):
        self._step()
        m = (1 << 52) - 1
        return ((int(self.x * (1 << 52)) ^ int(self.y * (1 << 52))) & m) / float(1 << 52)

    def randint(self, n):
        return min(n - 1, int(self.rand() * n))

    def pair(self):
        i = self.randint(SIZE)
        j = self.randint(SIZE - 1)
        return i, (j + 1 if j >= i else j)


# ----------------------------------------------------------------------------
def fwht(F):
    F = F.copy()
    m = F.shape[0]
    h = 1
    while h < SIZE:
        G = F.reshape(m, -1, 2 * h)
        a = G[:, :, :h].copy()
        b = G[:, :, h:].copy()
        G[:, :, :h] = a + b
        G[:, :, h:] = a - b
        F = G.reshape(m, SIZE)
        h *= 2
    return F


def walsh(s):
    return fwht(CHI[:, s])  # W[b,a]


def ddt(s):
    vals = s[XD] ^ s[None, :]
    return np.bincount((ROWOFF + vals).ravel(), minlength=(SIZE - 1) * SIZE).reshape(SIZE - 1, SIZE)


def swap_walsh(W, s, i, j):
    d = CHI[:, s[j]] - CHI[:, s[i]]
    return W + np.outer(d, HF[:, i] - HF[:, j])


def lex(W, D):
    aw = np.abs(W)
    wm = aw.max()
    du = D.max()
    return (int(SIZE // 2 - wm // 2), -int(du), -int((aw == wm).sum()), -int((D == du).sum()))


def cost_exp(W, D):
    return float(TW_EXP[np.abs(W).astype(np.int64)].sum() + TD_EXP[D].sum())


def cost_whs(W, D):
    return float(TW_WHS[np.abs(W).astype(np.int64)].sum())


class Tracker:
    """Keeps the best (lexicographic) S-box and the evaluation count."""
    def __init__(self, budget):
        self.budget, self.evals, self.best, self.best_lex = budget, 0, None, None
        self.trace = []

    def see(self, s, W, D):
        lx = lex(W, D)
        if self.best_lex is None or lx > self.best_lex:
            self.best_lex, self.best = lx, s.copy()

    def tick(self, k=1):
        self.evals += k
        if self.evals % 2000 < k and self.best_lex is not None:
            self.trace.append((self.evals, self.best_lex[0], -self.best_lex[1]))
        return self.evals < self.budget


# ----------------------------------------------------------------------------
def hc_lex(s0, rng, budget):
    t = Tracker(budget)
    s = s0.copy(); W = walsh(s); D = ddt(s); f = lex(W, D); t.see(s, W, D)
    while t.tick():
        i, j = rng.pair()
        s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
        W2 = swap_walsh(W, s, i, j); D2 = ddt(s2); f2 = lex(W2, D2)
        if f2 >= f:
            s, W, D, f = s2, W2, D2, f2
            t.see(s, W, D)
    return t


def sa(s0, rng, budget, costfn, use_ddt=True):
    t = Tracker(budget)
    s = s0.copy(); W = walsh(s); D = ddt(s); c = costfn(W, D); t.see(s, W, D)
    # T0 calibration: the average uphill move is accepted with probability 0.5
    ups = []
    for _ in range(200):
        t.tick()
        i, j = rng.pair()
        s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
        W2 = swap_walsh(W, s, i, j); D2 = ddt(s2) if use_ddt else D
        dc = costfn(W2, D2) - c
        if dc > 0:
            ups.append(dc)
    T = (np.mean(ups) if ups else 1.0) / math.log(2.0)
    n_left = budget - t.evals
    alpha = (1e-4) ** (1.0 / max(1, n_left))
    while t.tick():
        i, j = rng.pair()
        s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
        W2 = swap_walsh(W, s, i, j)
        D2 = ddt(s2)  # always needed for tracking the lexicographic key
        c2 = costfn(W2, D2)
        dc = c2 - c
        if dc <= 0 or rng.rand() < math.exp(-dc / T):
            s, W, D, c = s2, W2, D2, c2
            t.see(s, W, D)
        T *= alpha
    return t


def tabu(s0, rng, budget, costfn, K=16, tenure=24):
    t = Tracker(budget)
    s = s0.copy(); W = walsh(s); D = ddt(s); c = costfn(W, D); t.see(s, W, D)
    best_c = c
    tabu_until = np.zeros(SIZE, dtype=np.int64)
    it = 0
    while t.evals < budget:
        it += 1
        cand = None
        for _ in range(K):
            if not t.tick():
                break
            i, j = rng.pair()
            s2 = s.copy(); s2[i], s2[j] = s2[j], s2[i]
            W2 = swap_walsh(W, s, i, j); D2 = ddt(s2); c2 = costfn(W2, D2)
            is_tabu = tabu_until[i] > it or tabu_until[j] > it
            if is_tabu and c2 >= best_c:
                continue  # aspiration: a tabu move is allowed only if it beats the global best
            if cand is None or c2 < cand[0]:
                cand = (c2, i, j, s2, W2, D2)
        if cand is None:
            continue
        c, i, j, s, W, D = cand
        tabu_until[i] = tabu_until[j] = it + tenure
        best_c = min(best_c, c)
        t.see(s, W, D)
    return t


def cycle_crossover(p1, p2, rng):
    child = np.empty(SIZE, dtype=np.int64)
    filled = np.zeros(SIZE, dtype=bool)
    inv1 = np.empty(SIZE, dtype=np.int64); inv1[p1] = IDX
    take_first = rng.rand() < 0.5
    for start in range(SIZE):
        if filled[start]:
            continue
        idx = start
        while not filled[idx]:
            child[idx] = p1[idx] if take_first else p2[idx]
            filled[idx] = True
            idx = inv1[p2[idx]]
        take_first = not take_first
    return child


def ga(pop0, rng, budget, costfn, elite=2, tour=3, pm_swaps=3):
    t = Tracker(budget)
    pop = []
    for s in pop0:
        W = walsh(s); D = ddt(s); pop.append((costfn(W, D), s.copy())); t.see(s, W, D); t.tick()
    n = len(pop)

    def select():
        best = None
        for _ in range(tour):
            c = pop[rng.randint(n)]
            if best is None or c[0] < best[0]:
                best = c
        return best[1]

    while t.evals < budget:
        pop.sort(key=lambda z: z[0])
        new = pop[:elite]
        while len(new) < n and t.evals < budget:
            child = cycle_crossover(select(), select(), rng)
            for _ in range(1 + rng.randint(pm_swaps)):
                i, j = rng.pair()
                child[i], child[j] = child[j], child[i]
            W = walsh(child); D = ddt(child)
            new.append((costfn(W, D), child)); t.see(child, W, D); t.tick()
        pop = new
    return t


# ----------------------------------------------------------------------------
def job(args):
    method, seed_id, key, pop_keys, budget = args
    s0, _ = L.generate_sbox(P, *key)
    rng = ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    t0 = time.time()
    if method == "HC-lex":
        tr = hc_lex(s0, rng, budget)
    elif method == "SA-WHS":
        tr = sa(s0, rng, budget, cost_whs)
    elif method == "SA-EXP":
        tr = sa(s0, rng, budget, cost_exp)
    elif method == "TABU-EXP":
        tr = tabu(s0, rng, budget, cost_exp)
    elif method == "GA-EXP":
        pop0 = [s0] + [L.generate_sbox(P, *k)[0] for k in pop_keys]
        tr = ga(pop0, rng, budget, cost_exp)
    else:
        raise ValueError(method)
    return method, seed_id, tr.best, time.time() - t0, tr.evals, tr.trace


METHODS = ["HC-lex", "SA-WHS", "SA-EXP", "TABU-EXP", "GA-EXP"]


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    keys_rng = np.random.default_rng(7)
    seed_keys = [tuple(map(float, keys_rng.uniform(0.05, 0.95, 2))) for _ in range(n_seeds)]
    pop_keys = [[tuple(map(float, keys_rng.uniform(0.05, 0.95, 2))) for _ in range(15)] for _ in range(n_seeds)]

    # checks: FWHT == matrix product, incremental update == full recomputation
    s, _ = L.generate_sbox(P, *seed_keys[0])
    W = walsh(s)
    assert np.allclose(W, CHI[:, s] @ HF.T)
    s2 = s.copy(); s2[3], s2[77] = s2[77], s2[3]
    assert np.allclose(swap_walsh(W, s, 3, 77), walsh(s2))
    assert lex(W, ddt(s))[0] == L.analyze(s)["comp_NL"] and -lex(W, ddt(s))[1] == L.analyze(s)["DU"]

    jobs = [(m, k, seed_keys[k], pop_keys[k], budget) for m in METHODS for k in range(n_seeds)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"Butce: {budget} degerlendirme / calisma, {n_seeds} baslangic S-box'i, {len(jobs)} calisma")
    results = []
    with Pool(min(11, len(jobs))) as pool:
        for r in pool.imap_unordered(job, jobs):
            results.append(r)
            say(f"  bitti: {r[0]:<9} tohum {r[1]}  sure {r[3]:6.1f} s")

    say("\n" + L.HEADER)
    for k in range(n_seeds):
        s0, _ = L.generate_sbox(P, *seed_keys[k])
        say(L.row(f"baslangic t{k}", L.analyze(s0)))
    summary = {}
    for m in METHODS:
        for method, sid, sb, dt, ev, trace in sorted(results, key=lambda z: z[1]):
            if method != m:
                continue
            a = L.analyze(sb)
            summary.setdefault(m, []).append((a, sb, sid, dt, trace))
            say(L.row(f"{m} t{sid}", a))

    say("\n== Yontem ozeti (tohumlar uzerinden: en iyi / ortalama) ==")
    say(f"{'Yontem':<10} {'cNL eniyi':>9} {'cNL ort':>8} {'DU eniyi':>8} {'DU ort':>7} {'LP eniyi':>8} "
        f"{'NLkoord ort':>11} {'BIC-NL ort':>10} {'SAC ort':>8} {'BICSAC ort':>10} {'sure ort':>8}")
    overall = []
    for m in METHODS:
        rs = summary[m]
        cnl = [r[0]["comp_NL"] for r in rs]; du = [r[0]["DU"] for r in rs]; lp = [r[0]["LP"] for r in rs]
        say(f"{m:<10} {max(cnl):>9} {np.mean(cnl):>8.2f} {min(du):>8} {np.mean(du):>7.2f} {min(lp):>8.4f} "
            f"{np.mean([r[0]['NL_avg'] for r in rs]):>11.2f} {np.mean([r[0]['BIC_NL_avg'] for r in rs]):>10.2f} "
            f"{np.mean([r[0]['SAC_avg'] for r in rs]):>8.4f} {np.mean([r[0]['BIC_SAC_avg'] for r in rs]):>10.4f} "
            f"{np.mean([r[3] for r in rs]):>7.1f}s")
        overall.extend([(m,) + r for r in rs])

    say("\n== Yakinsama izleri (degerlendirme: bilesenNL/DU) ==")
    for m, a, sb, sid, dt, trace in overall:
        pts = trace[:: max(1, len(trace) // 8)] + trace[-1:]
        say(f"{m:<9} t{sid}: " + "  ".join(f"{e}:{nl}/{d}" for e, nl, d in pts))

    overall.sort(key=lambda z: (z[1]["comp_NL"], -z[1]["DU"], -z[1]["LP"], -z[1]["fixed_points"],
                                -abs(z[1]["SAC_avg"] - 0.5)), reverse=True)
    m, a, sb, sid, dt, trace = overall[0]
    say(f"\n== Genel en iyi S-box: {m}, tohum t{sid} (anahtar x0={seed_keys[sid][0]:.10f}, y0={seed_keys[sid][1]:.10f}) ==")
    say(L.HEADER)
    say(L.row(f"{m} t{sid}", a))
    say(L.fmt_sbox(sb))
    say(f"Koordinat NL: {a['NL_all']}")
    say(f"Cebirsel dereceler: {L.algebraic_degree(sb)}")
    say("SAC matrisi:")
    for rr in a["SAC_matrix"]:
        say("  " + " ".join(f"{v:.4f}" for v in rr))

    with open("lccm_optimizers_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
