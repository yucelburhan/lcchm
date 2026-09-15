"""
The remaining 3 NIST SP 800-22 tests: Non-overlapping Template (m=9), Overlapping Template (m=9), Linear Complexity (M=500).
Same 100 x 1e6-bit 2D-LCChM sequences (lccm_nist.gen_bits). Validation: NIST worked examples.
"""
import numpy as np
from math import sqrt, exp, log
from multiprocessing import Pool
import lccm_nist as N

SIZE_M = 9


def aperiodic_templates(m):
    out = []
    for v in range(1 << m):
        b = [(v >> (m - 1 - i)) & 1 for i in range(m)]
        ok = True
        for s in range(1, m):
            if b[:m - s] == b[s:]:
                ok = False; break
        if ok:
            out.append(v)
    return out


TEMPLATES = aperiodic_templates(SIZE_M)  # 148 templates


def windows(b, m):
    n = len(b) - m + 1
    w = np.zeros(n, dtype=np.int64)
    for j in range(m):
        w = (w << 1) | b[j:j + n]
    return w


def non_overlapping(b, m=SIZE_M, N_blocks=8):
    n = len(b); M = n // N_blocks
    mu = (M - m + 1) / 2 ** m
    var = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    pvals = []
    for t in TEMPLATES:
        W = np.array([(windows(b[i * M:(i + 1) * M], m) == t).sum() for i in range(N_blocks)], dtype=np.float64)
        chi = ((W - mu) ** 2 / var).sum()
        pvals.append(N.igamc(N_blocks / 2, chi / 2))
    return pvals


def overlapping(b, m=SIZE_M, M=1032):
    n = len(b); Nb = n // M
    lam = (M - m + 1) / 2 ** m; eta = lam / 2
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    tpl = (1 << m) - 1
    nu = np.zeros(6)
    for i in range(Nb):
        c = int((windows(b[i * M:(i + 1) * M], m) == tpl).sum())
        nu[min(c, 5)] += 1
    chi = ((nu - Nb * np.array(pi)) ** 2 / (Nb * np.array(pi))).sum()
    return N.igamc(2.5, chi / 2)


def berlekamp_massey(s):
    n = len(s); c = np.zeros(n, dtype=np.int8); bb = np.zeros(n, dtype=np.int8)
    c[0] = bb[0] = 1; L = 0; m = -1
    for i in range(n):
        d = (s[i] + int(np.dot(c[1:L + 1], s[i - L:i][::-1]) if L else 0)) & 1
        if d:
            t = c.copy()
            sh = i - m
            c[sh:] ^= bb[:n - sh]
            if L <= i // 2:
                L = i + 1 - L; m = i; bb = t
    return L


def linear_complexity(b, M=500):
    n = len(b); Nb = n // M
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / 2 ** M
    nu = np.zeros(7)
    for i in range(Nb):
        L = berlekamp_massey(b[i * M:(i + 1) * M].astype(np.int8))
        T = (-1) ** M * (L - mu) + 2 / 9
        if T <= -2.5: nu[0] += 1
        elif T <= -1.5: nu[1] += 1
        elif T <= -0.5: nu[2] += 1
        elif T <= 0.5: nu[3] += 1
        elif T <= 1.5: nu[4] += 1
        elif T <= 2.5: nu[5] += 1
        else: nu[6] += 1
    chi = ((nu - Nb * np.array(pi)) ** 2 / (Nb * np.array(pi))).sum()
    return N.igamc(3.0, chi / 2)


def job(seed):
    b = N.gen_bits(seed)
    return seed, non_overlapping(b), overlapping(b), linear_complexity(b)


def validate():
    lines = ["Dogrulama:"]
    # Non-overlapping: NIST example eps=10100100101110010110, B=001, M=10, N=2 -> 0.344154
    b = np.array([int(c) for c in "10100100101110010110"]); m = 3; M = 10; Nb = 2
    mu = (M - m + 1) / 2 ** m; var = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    W = np.array([(windows(b[i * M:(i + 1) * M], m) == 0b001).sum() for i in range(Nb)], dtype=np.float64)
    p = N.igamc(Nb / 2, ((W - mu) ** 2 / var).sum() / 2)
    lines.append(f"  NonOverlapping ornek: W={W.tolist()} p={p:.6f} (referans 0.344154) {'OK' if abs(p-0.344154)<1e-5 else 'FARKLI'}")
    # Linear complexity: NIST example 1101011110001 -> L=4; plus an LFSR sequence
    L = berlekamp_massey(np.array([int(c) for c in "1101011110001"], dtype=np.int8))
    lines.append(f"  Berlekamp-Massey ornek: L={L} (referans 4) {'OK' if L == 4 else 'FARKLI'}")
    # x^5+x^3+1 LFSR (m-sequence, L=5)
    st = [1, 0, 0, 1, 1]; seq = []
    for _ in range(200):
        seq.append(st[0]); nb = st[0] ^ st[2]; st = st[1:] + [nb]
    L2 = berlekamp_massey(np.array(seq, dtype=np.int8))
    lines.append(f"  LFSR x^5+x^3+1 dizisi: L={L2} (beklenen 5) {'OK' if L2 == 5 else 'FARKLI'}")
    lines.append(f"  Aperiyodik sablon sayisi (m=9): {len(TEMPLATES)} (referans 148) {'OK' if len(TEMPLATES) == 148 else 'FARKLI'}")
    ok = abs(p - 0.344154) < 1e-5 and L == 4 and L2 == 5 and len(TEMPLATES) == 148
    return ok, lines


if __name__ == "__main__":
    ok, lines = validate(); print("\n".join(lines)); assert ok
    with Pool(11) as pool:
        res = dict((s, r) for s, *r in pool.map(job, range(100)))
    n = 100; thr = 0.99 - 3 * sqrt(0.01 * 0.99 / n)
    lines.append(f"\n{'Test':<40} {'gecen':>7} {'oran':>6} {'p-dagilim':>10} {'sonuc':>6}")

    def evalp(ps, label):
        ps = np.array(ps); npass = (ps >= 0.01).sum()
        h = np.histogram(ps, bins=10, range=(0, 1))[0]
        pu = N.igamc(4.5, ((h - n / 10) ** 2 / (n / 10)).sum() / 2)
        good = npass / n >= thr and pu >= 1e-4
        lines.append(f"{label:<40} {npass:>3}/{n:<3} {npass/n:>6.3f} {pu:>10.4f} {'GECTI' if good else 'KALDI':>6}")
        return good

    allok = True
    fails = 0
    for ti, t in enumerate(TEMPLATES):
        ps = np.array([res[s][0][ti] for s in range(n)]); npass = (ps >= 0.01).sum()
        h = np.histogram(ps, bins=10, range=(0, 1))[0]
        pu = N.igamc(4.5, ((h - n / 10) ** 2 / (n / 10)).sum() / 2)
        if not (npass / n >= thr and pu >= 1e-4):
            fails += 1; lines.append(f"  NonOverlapping sablon {t:09b}: {npass}/100 oran {npass/n:.3f} p-dagilim {pu:.4f} KALDI")
    lines.append(f"{'NonOverlappingTemplate (148 sablon)':<40} {'':>7} {'':>6} {'':>10} {148 - fails:>3}/148 gecti")
    allok &= fails <= 3  # NIST: 1-3 chance failures are expected among the 148 templates
    allok &= evalp([res[s][1] for s in range(n)], "OverlappingTemplate (m=9)")
    allok &= evalp([res[s][2] for s in range(n)], "LinearComplexity (M=500)")
    lines.append(f"\nGenel: {'GECTI' if allok else 'KALDI'} (Non-overlapping icin 148 sablonda <=3 rastgele basarisizlik kabul edilir)")
    print("\n".join(lines[lines.index('') if '' in lines else 0:] if False else lines[-6:]))
    open("lccm_nist_rest_results.txt", "w", encoding="utf-8").write("\n".join(lines))
