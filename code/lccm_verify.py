"""
Independent verification: every S-box metric is computed by brute force FROM ITS DEFINITION.
No function of lccm_sbox / lccm_optimizers is used (no Walsh transform, no incremental update).
The results are compared with lccm_sbox.analyze.

Usage: python lccm_verify.py <sbox.npy> [<sbox2.npy> ...]
"""
import sys
import numpy as np

N, SZ = 8, 256


def parity(v):
    v = np.asarray(v, dtype=np.int64)
    p = np.zeros_like(v)
    while v.any():
        p ^= v & 1
        v = v >> 1
    return p


def verify(S):
    S = [int(v) for v in S]
    Sa = np.array(S, dtype=np.int64)
    xs = np.arange(SZ, dtype=np.int64)
    r = {}
    r["bijective"] = sorted(S) == list(range(SZ))

    # NL and LP from the definition: agreement count #{x : b.S(x) == a.x}
    PA = parity(xs[:, None] & xs[None, :])            # [a, x]
    PB = parity(np.arange(1, SZ)[:, None] & Sa[None, :])  # [b-1, x]
    agree = np.zeros((SZ - 1, SZ), dtype=np.int64)
    for b in range(SZ - 1):
        agree[b] = (PB[b][None, :] == PA).sum(axis=1)
    dist = np.minimum(agree, SZ - agree)              # distance to the affine function (a.x or a.x+1)
    nl_b = dist.min(axis=1)                            # NL of component b
    r["comp_NL"] = int(nl_b.min())
    coord = [int(nl_b[(1 << i) - 1]) for i in range(N)]
    r["NL_all"] = coord
    r["NL_min"], r["NL_avg"], r["NL_max"] = min(coord), sum(coord) / N, max(coord)
    bic_nl = [int(nl_b[((1 << i) | (1 << j)) - 1]) for i in range(N) for j in range(i + 1, N)]
    r["BIC_NL_min"], r["BIC_NL_avg"] = min(bic_nl), sum(bic_nl) / len(bic_nl)
    bias = np.abs(agree[:, 1:] - SZ // 2)              # a != 0
    bias0 = np.abs(agree[:, 0] - SZ // 2)
    r["LP"] = max(int(bias.max()), int(bias0.max())) / SZ

    # DDT and DU (plain loops)
    du = 0
    for dx in range(1, SZ):
        cnt = [0] * SZ
        for x in range(SZ):
            cnt[S[x] ^ S[x ^ dx]] += 1
        du = max(du, max(cnt))
    r["DU"], r["DP"] = du, du / SZ

    # SAC (plain loops)
    M = [[0.0] * N for _ in range(N)]
    for k in range(N):
        for i in range(N):
            c = sum(((S[x] ^ S[x ^ (1 << k)]) >> i) & 1 for x in range(SZ))
            M[i][k] = c / SZ
    flat = [v for row in M for v in row]
    r["SAC_avg"], r["SAC_min"], r["SAC_max"] = sum(flat) / len(flat), min(flat), max(flat)

    # BIC-SAC (plain loops)
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(N):
                c = 0
                for x in range(SZ):
                    y1, y2 = S[x], S[x ^ (1 << k)]
                    c += (((y1 >> i) ^ (y1 >> j)) ^ ((y2 >> i) ^ (y2 >> j))) & 1
                vals.append(c / SZ)
    r["BIC_SAC_avg"] = sum(vals) / len(vals)

    # fixed points, algebraic degree (ANF: coefficient a_u = XOR over subsets x of u of f(x))
    r["fixed_points"] = sum(1 for x in range(SZ) if S[x] == x)
    degs = []
    for i in range(N):
        f = [(S[x] >> i) & 1 for x in range(SZ)]
        d = 0
        for u in range(SZ):
            a = 0
            sub = u
            while True:
                a ^= f[sub]
                if sub == 0:
                    break
                sub = (sub - 1) & u
            if a:
                d = max(d, bin(u).count("1"))
        degs.append(d)
    r["degrees"] = degs
    return r


def aes():
    def gm(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return p
    inv = [0] + [next(y for y in range(1, 256) if gm(x, y) == 1) for x in range(1, 256)]
    out = []
    for x in range(256):
        b = inv[x]
        s = b
        for k in range(1, 5):
            s ^= ((b << k) | (b >> (8 - k))) & 0xFF
        out.append(s ^ 0x63)
    return out


if __name__ == "__main__":
    import lccm_sbox as L
    keys = ["bijective", "comp_NL", "NL_min", "NL_avg", "NL_max", "BIC_NL_min", "BIC_NL_avg", "SAC_avg",
            "SAC_min", "SAC_max", "BIC_SAC_avg", "LP", "DP", "DU", "fixed_points"]
    targets = [("AES", np.array(aes()))] + [(p, np.load(p).astype(np.int64)) for p in sys.argv[1:]]
    lines = []
    for name, S in targets:
        v = verify(S)
        a = L.analyze(S)
        a["degrees"] = L.algebraic_degree(S)
        lines.append(f"== {name} ==")
        ok_all = True
        for k in keys + ["NL_all", "degrees"]:
            av, vv = a[k], v[k]
            ok = (abs(av - vv) < 1e-12) if isinstance(vv, float) else (list(av) == list(vv) if isinstance(vv, list) else av == vv)
            ok_all &= bool(ok)
            lines.append(f"  {k:<13} bagimsiz={vv!s:<32} analiz={av!s:<32} {'OK' if ok else 'FARKLI'}")
        lines.append(f"  SONUC: {'TUM OLCUTLER ESLESIYOR' if ok_all else 'UYUSMAZLIK VAR'}")
    print("\n".join(lines))
    open("lccm_verify_results.txt", "w", encoding="utf-8").write("\n".join(lines))
