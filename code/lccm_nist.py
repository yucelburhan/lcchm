"""
NIST SP 800-22 (12 of the 15 tests; the remaining three are in lccm_nist_rest.py) on the 2D-LCChM byte stream.
Self-validation against the worked examples of the NIST document.

Sequences: 100 x 1e6 bits. Byte per iteration = (int(x*1e14) ^ int(y*1e14)) & 0xFF (same as S-box generation).
Pass criteria: proportion >= 0.9602 (100 sequences), p-value uniformity chi2 (10 bins) p >= 1e-4.
"""
import math
import numpy as np
from math import erfc, sqrt, log, exp
from multiprocessing import Pool

import lccm_scan_beta as S
import lccm_sbox as L

P = L.load_params({"mu": 3.99, "rho": 2.59, "k": 5.3, "lam": 0.7, "beta": 1.0})


def igamc(a, x):
    """Upper regularised incomplete gamma Q(a,x) (series / continued fraction, as in the NIST reference code)."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        # series
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * exp(-x + a * log(x) - math.lgamma(a))
    # continued fraction (Lentz)
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if d == 0 else d
        c = b + an / c; c = 1e-300 if c == 0 else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-15:
            break
    return exp(-x + a * log(x) - math.lgamma(a)) * h


def frequency(b):
    n = len(b); s = 2 * b.sum() - n
    return erfc(abs(s) / sqrt(n) / sqrt(2))


def block_frequency(b, M=128):
    n = len(b); N = n // M
    pi = b[:N * M].reshape(N, M).mean(axis=1)
    chi = 4 * M * ((pi - 0.5) ** 2).sum()
    return igamc(N / 2, chi / 2)


def runs(b):
    n = len(b); pi = b.mean()
    if abs(pi - 0.5) >= 2 / sqrt(n):
        return 0.0
    v = 1 + (b[1:] != b[:-1]).sum()
    return erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * sqrt(2 * n) * pi * (1 - pi)))


def longest_run(b):
    n = len(b)
    if n >= 750000:
        M, K, vals, pi = 10000, 6, [10, 11, 12, 13, 14, 15, 16], [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    elif n >= 6272:
        M, K, vals, pi = 128, 5, [4, 5, 6, 7, 8, 9], [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K, vals, pi = 8, 3, [1, 2, 3, 4], [0.2148, 0.3672, 0.2305, 0.1875]
    N = n // M
    blocks = b[:N * M].reshape(N, M)
    # longest run of ones in each block
    longest = np.zeros(N, dtype=np.int64); cur = np.zeros(N, dtype=np.int64)
    for j in range(M):
        cur = (cur + 1) * blocks[:, j]
        longest = np.maximum(longest, cur)
    nu = np.zeros(K + 1)
    for i, lo in enumerate(vals):
        if i == 0:
            nu[i] = (longest <= lo).sum()
        elif i == K:
            nu[i] = (longest >= lo).sum()
        else:
            nu[i] = (longest == lo).sum()
    chi = ((nu - N * np.array(pi)) ** 2 / (N * np.array(pi))).sum()
    return igamc(K / 2, chi / 2)


def gf2_rank(M):
    M = M.copy(); rows, cols = M.shape; r = 0
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        M[others] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def rank_test(b, Mr=32, Q=32):
    n = len(b); N = n // (Mr * Q)
    ranks = np.array([gf2_rank(b[i * Mr * Q:(i + 1) * Mr * Q].reshape(Mr, Q)) for i in range(N)])
    F32 = (ranks == 32).sum(); F31 = (ranks == 31).sum(); F30 = N - F32 - F31
    p32, p31, p30 = 0.2888, 0.5776, 0.1336
    chi = (F32 - p32 * N) ** 2 / (p32 * N) + (F31 - p31 * N) ** 2 / (p31 * N) + (F30 - p30 * N) ** 2 / (p30 * N)
    return exp(-chi / 2)


def dft_test(b):
    n = len(b); x = 2 * b.astype(np.float64) - 1
    mod = np.abs(np.fft.fft(x))[: n // 2]
    T = sqrt(log(1 / 0.05) * n)
    N0 = 0.95 * n / 2; N1 = (mod < T).sum()
    d = (N1 - N0) / sqrt(n * 0.95 * 0.05 / 4)
    return erfc(abs(d) / sqrt(2))


def universal(b, Lb=7, Q=1280):
    n = len(b); K = n // Lb - Q
    ev, var = {6: (5.2177052, 2.954), 7: (6.1962507, 3.125), 8: (7.1836656, 3.238)}[Lb]
    blocks = b[:(Q + K) * Lb].reshape(Q + K, Lb) @ (1 << np.arange(Lb - 1, -1, -1))
    T = np.zeros(1 << Lb, dtype=np.int64)
    for i in range(Q):
        T[blocks[i]] = i + 1
    s = 0.0
    for i in range(Q, Q + K):
        s += log(i + 1 - T[blocks[i]], 2); T[blocks[i]] = i + 1
    fn = s / K
    c = 0.7 - 0.8 / Lb + (4 + 32 / Lb) * K ** (-3 / Lb) / 15
    sigma = c * sqrt(var / K)
    return erfc(abs(fn - ev) / (sqrt(2) * sigma))


def _psi2(b, m):
    n = len(b)
    if m == 0:
        return 0.0
    ext = np.concatenate([b, b[: m - 1]])
    w = np.zeros(n, dtype=np.int64)
    for j in range(m):
        w = (w << 1) | ext[j:j + n]
    cnt = np.bincount(w, minlength=1 << m)
    return (cnt.astype(np.float64) ** 2).sum() * (1 << m) / n - n


def serial(b, m=16):
    n = len(b)
    d1 = _psi2(b, m) - _psi2(b, m - 1); d2 = _psi2(b, m) - 2 * _psi2(b, m - 1) + _psi2(b, m - 2)
    return igamc(2 ** (m - 2), d1 / 2), igamc(2 ** (m - 3), d2 / 2)


def _phi(b, m):
    n = len(b); ext = np.concatenate([b, b[: m - 1]])
    w = np.zeros(n, dtype=np.int64)
    for j in range(m):
        w = (w << 1) | ext[j:j + n]
    cnt = np.bincount(w, minlength=1 << m).astype(np.float64)
    c = cnt[cnt > 0] / n
    return (c * np.log(c)).sum()


def approx_entropy(b, m=10):
    n = len(b); ap = _phi(b, m) - _phi(b, m + 1)
    chi = 2 * n * (log(2) - ap)
    return igamc(2 ** (m - 1), chi / 2)


def cusum(b, backward=False):
    n = len(b); x = 2 * b.astype(np.int64) - 1
    if backward:
        x = x[::-1]
    z = np.abs(np.cumsum(x)).max()
    from math import floor
    from statistics import NormalDist
    Phi = NormalDist().cdf
    s = 0.0
    for k in range(floor((-n / z + 1) / 4), floor((n / z - 1) / 4) + 1):
        s += Phi((4 * k + 1) * z / sqrt(n)) - Phi((4 * k - 1) * z / sqrt(n))
    t = 0.0
    for k in range(floor((-n / z - 3) / 4), floor((n / z - 1) / 4) + 1):
        t += Phi((4 * k + 3) * z / sqrt(n)) - Phi((4 * k + 1) * z / sqrt(n))
    return 1 - s + t


def random_excursions(b):
    x = 2 * b.astype(np.int64) - 1
    s = np.concatenate([[0], np.cumsum(x), [0]])
    zeros = np.nonzero(s == 0)[0]
    J = len(zeros) - 1
    if J < 500:
        return None, None
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    pi = {1: [0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0312], 2: [0.75, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
          3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804], 4: [0.875, 0.0156, 0.0137, 0.012, 0.0105, 0.0733]}
    # number of visits to each state in each cycle
    cyc = np.cumsum(s == 0)[:-1]  # cycle index
    pv = []
    for st in states:
        vis = np.bincount(cyc[s[:-1] == st], minlength=J + 1)[1:J + 1]
        nu = [((vis == k) if k < 5 else (vis >= 5)).sum() for k in range(6)]
        pr = pi[abs(st)]
        chi = sum((nu[k] - J * pr[k]) ** 2 / (J * pr[k]) for k in range(6))
        pv.append(igamc(2.5, chi / 2))
    # variant
    pvv = []
    for st in list(range(-9, 0)) + list(range(1, 10)):
        xi = (s == st).sum()
        pvv.append(erfc(abs(xi - J) / sqrt(2 * J * (4 * abs(st) - 2))))
    return pv, pvv


TESTS = ["Frequency", "BlockFrequency", "Runs", "LongestRun", "Rank", "DFT", "Universal", "Serial-1", "Serial-2",
         "ApproxEntropy", "CuSum-fwd", "CuSum-bwd", "RandomExcursions", "RandExcVariant"]


def all_tests(b):
    r = {"Frequency": frequency(b), "BlockFrequency": block_frequency(b), "Runs": runs(b), "LongestRun": longest_run(b),
         "Rank": rank_test(b), "DFT": dft_test(b), "Universal": universal(b)}
    r["Serial-1"], r["Serial-2"] = serial(b)
    r["ApproxEntropy"] = approx_entropy(b); r["CuSum-fwd"] = cusum(b); r["CuSum-bwd"] = cusum(b, True)
    re, rv = random_excursions(b)
    r["RandomExcursions"] = min(re) if re else None  # minimum over the 8 states (conservative; per-state evaluation in lccm_nist_re.py)
    r["RandExcVariant"] = min(rv) if rv else None
    return r


def gen_bits(seed, nbits=1_000_000):
    g = np.random.default_rng(seed)
    x0, y0 = g.uniform(0.05, 0.95, 2)
    x, y = x0, y0
    for n in range(1000):
        x, y = L.lccm_step(x, y, n, P)
    nbytes = nbits // 8
    out = np.empty(nbytes, dtype=np.uint8)
    for i in range(nbytes):
        x, y = L.lccm_step(x, y, 1000 + i, P)
        out[i] = (int(x * 1e14) ^ int(y * 1e14)) & 0xFF
    return np.unpackbits(out)[:nbits].astype(np.int64)


def job(seed):
    return seed, all_tests(gen_bits(seed))


def validate():
    eps = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
    b = np.array([int(c) for c in eps])
    ref = {"Frequency": (frequency(b), 0.109599), "BlockFrequency(M=10)": (block_frequency(b, 10), 0.706438),
           "Runs": (runs(b), 0.500798), "CuSum-fwd": (cusum(b), 0.219194), "CuSum-bwd": (cusum(b, True), 0.114866),
           "ApproxEntropy(m=2)": (approx_entropy(b, 2), 0.235301)}
    # Serial test: 10-bit NIST example (0011011101, m=3). The 0.9057/0.8805 printed in the document is a known erratum;
    # the sts-2.1.2 code gives 0.808792 / 0.670320 (psi2 = 2.8, 1.2, 0.4 as in the document).
    b10 = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    s1, s2 = serial(b10, 3)
    ref["Serial(m=3)-1"] = (s1, 0.808792); ref["Serial(m=3)-2"] = (s2, 0.670320)
    ref["Serial psi2_3"] = (_psi2(b10, 3), 2.8)
    lines = ["Test vektoru dogrulamasi (NIST SP 800-22 ornekleri; seri test sts-2.1.2 degerleri):"]
    ok = True
    for k, (got, exp_) in ref.items():
        good = abs(got - exp_) < 5e-4
        ok &= good
        lines.append(f"  {k:<22} hesap={got:.6f} referans={exp_:.6f} {'OK' if good else 'FARKLI'}")
    return ok, lines


if __name__ == "__main__":
    ok, lines = validate()
    print("\n".join(lines))
    assert ok, "NIST dogrulamasi basarisiz"
    with Pool(11) as pool:
        res = dict(pool.map(job, range(100)))
    lines.append(f"\n2D-LCCM parametreleri: {P}")
    lines.append(f"{'Test':<18} {'gecen/100':>9} {'oran':>6} {'p-dagilim p':>12} {'sonuc':>7}")
    allpass = True
    for t in TESTS:
        ps = np.array([res[s][t] for s in range(100) if res[s][t] is not None])
        if len(ps) == 0:
            lines.append(f"{t:<18} uygulanamadi"); continue
        npass = (ps >= 0.01).sum()
        h = np.histogram(ps, bins=10, range=(0, 1))[0]
        chi = ((h - len(ps) / 10) ** 2 / (len(ps) / 10)).sum()
        pu = igamc(4.5, chi / 2)
        thr = 0.99 - 3 * sqrt(0.01 * 0.99 / len(ps))
        good = npass / len(ps) >= thr and pu >= 1e-4
        allpass &= good
        lines.append(f"{t:<18} {npass:>5}/{len(ps):<3} {npass/len(ps):>6.3f} {pu:>12.4f} {'GECTI' if good else 'KALDI'}")
    lines.append(f"\nGenel: {'TUM TESTLER GECTI' if allpass else 'BAZI TESTLER KALDI'} (oran esigi ~0.9602, dagilim esigi 1e-4)")
    print("\n".join(lines[len(lines) - len(TESTS) - 4:]))
    open("lccm_nist_results.txt", "w", encoding="utf-8").write("\n".join(lines))
