"""
Walsh-guided tabu search: raises the component nonlinearity to 104.

Change of a Walsh coefficient under the swap (i,j):
  dW[b,a] = (chi_b(S(j)) - chi_b(S(i))) * (H[a,i] - H[a,j])
It is non-zero only if c_i = chi_b(S(i))H[a,i] equals c_j and chi_b(S(i)) != chi_b(S(j)); then the
sign-adjusted change of |W| is -4*C_i (C = sign(W)*c). Hence the score "how many critical coefficients
decrease / increase" is obtained for ALL swaps with a single matrix product.
The K best candidates are evaluated exactly; the best one is accepted (even if worse) under the tabu rule.
All random decisions come from the 2D-LCChM stream.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import numpy as np

import lccm_sbox as L
import lccm_optimizers as O

SIZE = 256
IDX = np.arange(SIZE)
HF = L.HF
CHI = O.CHI  # (255,256) chi[b-1, y]
TRIU = np.triu(np.ones((SIZE, SIZE), dtype=bool), 1)


def key_of(W):
    aw = np.abs(W)
    m = aw.max()
    return (int(SIZE // 2 - m // 2), -int((aw == m).sum()), -int((aw == m - 4).sum()))


def pair_scores(s, W):
    aw = np.abs(W)
    m = aw.max()
    rows_m, cols_m = np.nonzero(aw == m)
    rows_n, cols_n = np.nonzero(aw == m - 4)

    def mats(rows, cols):
        chi = CHI[rows][:, s]                       # (k,256) chi_b(S(x))
        h = HF[cols]                                # (k,256) H[a,x]
        sg = np.sign(W[rows, cols])[:, None]
        C = sg * chi * h
        Pp = ((C > 0) & (chi > 0)).astype(np.float64)
        Pm = ((C > 0) & (chi < 0)).astype(np.float64)
        Np = ((C < 0) & (chi > 0)).astype(np.float64)
        Nm = ((C < 0) & (chi < 0)).astype(np.float64)
        dec = Pp.T @ Pm; dec = dec + dec.T          # |W| decreases by 4
        inc = Np.T @ Nm; inc = inc + inc.T          # |W| increases by 4
        return dec, inc

    dec_m, inc_m = mats(rows_m, cols_m)
    score = 1.0 * dec_m - 8.0 * inc_m
    if len(rows_n):
        dec_n, inc_n = mats(rows_n, cols_n)
        score += 0.02 * dec_n - 1.0 * inc_n
    return score


def guided_tabu(s0, rng, max_iter, target=104, K=24, tenure=12, log_every=0):
    s = s0.copy()
    W = O.walsh(s)
    k = key_of(W)
    best_k, best_s = k, s.copy()
    tabu_until = np.zeros(SIZE, dtype=np.int64)
    hit = None
    for it in range(1, max_iter + 1):
        sc = pair_scores(s, W)
        # exclude tabu positions; break ties with small chaotic noise
        tabu_mask = tabu_until > it
        sc[tabu_mask, :] = -1e18
        sc[:, tabu_mask] = -1e18
        sc = np.where(TRIU, sc, -1e18)
        flat = np.argpartition(sc.ravel(), -4 * K)[-4 * K:]
        noise = np.array([rng.rand() for _ in range(len(flat))]) * 1e-3
        flat = flat[np.argsort(-(sc.ravel()[flat] + noise))][:K]
        cand = None
        for f in flat:
            i, j = divmod(int(f), SIZE)
            W2 = O.swap_walsh(W, s, i, j)
            k2 = key_of(W2)
            if cand is None or k2 > cand[0]:
                cand = (k2, i, j, W2)
        k2, i, j, W2 = cand
        s[i], s[j] = s[j], s[i]
        W, k = W2, k2
        tabu_until[i] = tabu_until[j] = it + tenure
        if k > best_k:
            best_k, best_s = k, s.copy()
            if k[0] >= target and hit is None:
                hit = it
                break
        if log_every and it % log_every == 0:
            print(f"    it {it}: simdiki {k}  en iyi {best_k}", flush=True)
    return best_s, best_k, hit


if __name__ == "__main__":
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    s0, _ = L.generate_sbox(O.P, 0.6125859199, 0.8574924209)
    rng = O.ChaosRNG(0.53, 0.61)
    t0 = time.time()
    s, k, hit = guided_tabu(s0, rng, max_iter, log_every=250)
    a = L.analyze(s)
    print(f"sonuc: {k}, hedef@{hit}, {time.time() - t0:.0f}s, cNL={a['comp_NL']} DU={a['DU']} LP={a['LP']:.4f}")
    np.save("guided_test.npy", s)
