"""
2D-LCChM: 2D Logistic-Cubic-Chebyshev chaotic map, S-box generation and analysis.

Map (n: iteration index, theta_n = 2*pi*phi*n, phi = (sqrt5-1)/2 irrational ->
the modulation is quasi-periodic, not periodic):

  x_{n+1} = [ mu*x_n*(1-x_n)                      (logistic)
            + beta*T_k(2*y_n - 1)                 (Chebyshev, cross-coupled from y)
            + lam*sin(pi*(x_n + y_n) + theta_n) ] mod 1

  y_{n+1} = [ rho*y_n*(1-y_n^2)                   (cubic)
            + beta*T_k(2*x_{n+1} - 1)             (Chebyshev, cross-coupled from x)
            + lam*cos(pi*x_{n+1}*y_n + theta_n) ] mod 1

  T_k(u) = cos(k*arccos(u)),  u in [-1, 1];  beta = 1 in the working regime
"""
import math
import sys
import numpy as np

PHI = (math.sqrt(5.0) - 1.0) / 2.0


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
def cheb(k, u):
    u = min(1.0, max(-1.0, u))
    return math.cos(k * math.acos(u))


def lccm_step(x, y, n, p):
    mu, rho, k, lam = p["mu"], p["rho"], p["k"], p["lam"]
    beta = p.get("beta", 1.0)  # weight of the Chebyshev term (working regime: 1.0)
    th = 2.0 * math.pi * PHI * n
    xn = (mu * x * (1 - x) + beta * cheb(k, 2 * y - 1) + lam * math.sin(math.pi * (x + y) + th)) % 1.0
    yn = (rho * y * (1 - y * y) + beta * cheb(k, 2 * xn - 1) + lam * math.cos(math.pi * xn * y + th)) % 1.0
    return xn, yn


def load_params(default):
    """Read the map parameters from lccm_config.json if it exists (alternative regimes)."""
    import json, os
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lccm_config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            d = json.load(f)
        return {**default, **d}
    return dict(default)


def lyapunov(p, x0=0.31, y0=0.67, n_iter=100000, discard=1000):
    """Both Lyapunov exponents by the QR method."""
    mu, rho, k, lam = p["mu"], p["rho"], p["k"], p["lam"]

    def dcheb(u):
        u = min(1 - 1e-12, max(-1 + 1e-12, u))
        return k * math.sin(k * math.acos(u)) / math.sqrt(1 - u * u)

    x, y = x0, y0
    for n in range(discard):
        x, y = lccm_step(x, y, n, p)
    Q = np.eye(2)
    s = np.zeros(2)
    for n in range(discard, discard + n_iter):
        th = 2 * math.pi * PHI * n
        a = math.pi * (x + y) + th
        dxx = mu * (1 - 2 * x) + lam * math.pi * math.cos(a)
        dxy = 2 * dcheb(2 * y - 1) + lam * math.pi * math.cos(a)
        xn = (mu * x * (1 - x) + cheb(k, 2 * y - 1) + lam * math.sin(a)) % 1.0
        b = math.pi * xn * y + th
        dyxn = 2 * dcheb(2 * xn - 1) - lam * math.pi * y * math.sin(b)
        dyy_dir = rho * (1 - 3 * y * y) - lam * math.pi * xn * math.sin(b)
        yn = (rho * y * (1 - y * y) + cheb(k, 2 * xn - 1) + lam * math.cos(b)) % 1.0
        J = np.array([[dxx, dxy], [dyxn * dxx, dyy_dir + dyxn * dxy]])
        Q, R = np.linalg.qr(J @ Q)
        s += np.log(np.abs(np.diag(R)) + 1e-300)
        x, y = xn, yn
    return s / n_iter


# ----------------------------------------------------------------------------
# S-box generation
# ----------------------------------------------------------------------------
def generate_sbox(p, x0, y0, discard=1000):
    x, y = x0, y0
    n = 0
    for _ in range(discard):
        x, y = lccm_step(x, y, n, p)
        n += 1
    seen = [False] * 256
    sbox = []
    while len(sbox) < 256:
        x, y = lccm_step(x, y, n, p)
        n += 1
        v = (int(x * 1e14) ^ int(y * 1e14)) % 256
        if not seen[v]:
            seen[v] = True
            sbox.append(v)
    return np.array(sbox, dtype=np.int64), (x, y, n)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
N = 8
SIZE = 256
IDX = np.arange(SIZE)
POP = np.array([bin(i).count("1") for i in range(SIZE)])
# Hadamard: H[a,x] = (-1)^(a.x)
H = np.where(POP[IDX[:, None] & IDX[None, :]] % 2 == 0, 1, -1).astype(np.int64)


def bits(sbox):
    return ((sbox[None, :] >> np.arange(N)[:, None]) & 1)  # (8,256)


def walsh_max(fbits):
    """fbits: (m,256) 0/1 truth tables -> max|W| of each function."""
    W = (1 - 2 * fbits) @ H.T
    return np.abs(W).max(axis=1)


def nonlinearity(fbits):
    return (SIZE // 2) - walsh_max(fbits) // 2


def sac_matrix(sbox):
    M = np.zeros((N, N))
    for k in range(N):
        d = sbox ^ sbox[IDX ^ (1 << k)]
        M[:, k] = bits(d).mean(axis=1)
    return M  # M[i,k]: probability that output bit i flips when input bit k is complemented


def bic(sbox):
    B = bits(sbox)
    nls, sacs = [], []
    for i in range(N):
        for j in range(i + 1, N):
            f = B[i] ^ B[j]
            nls.append(int(nonlinearity(f[None, :])[0]))
            for k in range(N):
                sacs.append((f ^ f[IDX ^ (1 << k)]).mean())
    return np.array(nls), np.array(sacs)


def lp_dp(sbox):
    # all component functions b.S(x), b != 0
    comp = POP[IDX[1:, None] & sbox[None, :]] % 2  # (255,256)
    W = (1 - 2 * comp) @ H.T  # (255,256), column a=0 included
    W[:, 0] = 0
    lp_max = np.abs(W).max() / 2
    comp_nl = SIZE // 2 - lp_max
    LP = lp_max / SIZE
    DDT_max = 0
    for dx in range(1, SIZE):
        cnt = np.bincount(sbox ^ sbox[IDX ^ dx], minlength=SIZE)
        DDT_max = max(DDT_max, cnt.max())
    return LP, int(comp_nl), int(DDT_max)


def algebraic_degree(sbox):
    B = bits(sbox).copy()
    degs = []
    for i in range(N):
        a = B[i].copy()
        for k in range(N):  # Moebius transform
            step = 1 << k
            for x in range(SIZE):
                if x & step:
                    a[x] ^= a[x ^ step]
        degs.append(int(POP[a == 1].max()) if a.any() else 0)
    return degs


def analyze(sbox):
    B = bits(sbox)
    nl = nonlinearity(B)
    M = sac_matrix(sbox)
    bnl, bsac = bic(sbox)
    LP, comp_nl, du = lp_dp(sbox)
    return {
        "bijective": len(set(sbox.tolist())) == 256,
        "NL_min": int(nl.min()), "NL_avg": float(nl.mean()), "NL_max": int(nl.max()),
        "NL_all": nl.tolist(),
        "SAC_avg": float(M.mean()), "SAC_min": float(M.min()), "SAC_max": float(M.max()),
        "SAC_offset": float(np.abs(M - 0.5).mean()),
        "BIC_NL_min": int(bnl.min()), "BIC_NL_avg": float(bnl.mean()),
        "BIC_SAC_avg": float(bsac.mean()), "BIC_SAC_min": float(bsac.min()), "BIC_SAC_max": float(bsac.max()),
        "LP": LP, "comp_NL": comp_nl, "DU": du, "DP": du / 256,
        "fixed_points": int((sbox == IDX).sum()),
        "SAC_matrix": M,
    }


# ----------------------------------------------------------------------------
# Chaos-driven optimisation (swap-based hill climbing)
# ----------------------------------------------------------------------------
def fitness(sbox):
    B = bits(sbox)
    wm = walsh_max(B)
    nl = SIZE // 2 - wm // 2
    # primary: min NL, secondary: sum of NL, tertiary: reduce large Walsh coefficients
    W = np.abs((1 - 2 * B) @ H.T)
    return (int(nl.min()), int(nl.sum()), -int((W.astype(np.float64) ** 4).sum() // 1e6))


def optimize(sbox, p, state, iters=6000):
    x, y, n = state
    best = sbox.copy()
    bf = fitness(best)
    for _ in range(iters):
        x, y = lccm_step(x, y, n, p); n += 1
        i = int(x * 1e14) % 256
        j = int(y * 1e14) % 256
        if i == j:
            continue
        cand = best.copy()
        cand[i], cand[j] = cand[j], cand[i]
        cf = fitness(cand)
        if cf >= bf:
            best, bf = cand, cf
    return best


HF = H.astype(np.float64)
COMP_MASK = POP[IDX[1:, None] & IDX[None, :]] % 2  # (255,256): parity table of b.y


def fitness_full(sbox):
    """Over all 255 component functions: (min NL, -DU, -count of max|W|, -count of max DDT)."""
    comp = COMP_MASK[:, sbox]  # b.S(x)
    W = np.abs((1.0 - 2.0 * comp) @ HF.T)
    W[:, 0] = 0
    wm = W.max()
    ddt = np.zeros((SIZE - 1, SIZE), dtype=np.int64)
    for dx in range(1, SIZE):
        ddt[dx - 1] = np.bincount(sbox ^ sbox[IDX ^ dx], minlength=SIZE)
    du = ddt.max()
    return (int(SIZE // 2 - wm // 2), -int(du), -int((W == wm).sum()), -int((ddt == du).sum()))


def optimize_full(sbox, p, state, iters=4000):
    x, y, n = state
    best = sbox.copy()
    bf = fitness_full(best)
    for _ in range(iters):
        x, y = lccm_step(x, y, n, p); n += 1
        i = int(x * 1e14) % 256
        j = int(y * 1e14) % 256
        if i == j:
            continue
        cand = best.copy()
        cand[i], cand[j] = cand[j], cand[i]
        cf = fitness_full(cand)
        if cf >= bf:
            best, bf = cand, cf
    return best


# ----------------------------------------------------------------------------
def aes_sbox():
    def gmul(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11B
            b >>= 1
        return r
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if gmul(a, b) == 1:
                inv[a] = b
                break
    s = []
    for a in range(256):
        b = inv[a]
        r = b
        for sh in range(1, 5):
            r ^= ((b << sh) | (b >> (8 - sh))) & 0xFF
        s.append(r ^ 0x63)
    return np.array(s, dtype=np.int64)


def fmt_sbox(s):
    lines = []
    for r in range(16):
        lines.append(" ".join(f"{v:3d}" for v in s[r * 16:(r + 1) * 16]))
    return "\n".join(lines)


def row(name, r):
    return (f"{name:<22} {r['NL_min']:>4} {r['NL_avg']:>7.2f} {r['NL_max']:>4} "
            f"{r['SAC_avg']:>7.4f} {r['SAC_min']:>6.4f} {r['SAC_max']:>6.4f} "
            f"{r['BIC_NL_min']:>5} {r['BIC_NL_avg']:>7.2f} {r['BIC_SAC_avg']:>7.4f} "
            f"{r['LP']:>7.4f} {r['DP']:>7.4f} ({r['DU']:>2}) {r['fixed_points']:>3} {r['comp_NL']:>5}")


HEADER = (f"{'S-box':<22} {'NLmn':>4} {'NLavg':>7} {'NLmx':>4} {'SACavg':>7} {'SACmn':>6} {'SACmx':>6} "
          f"{'BNLmn':>5} {'BNLavg':>7} {'BSAC':>7} {'LP':>7} {'DP':>7} {'DU':>4} {'FP':>3} {'cNL':>5}")


if __name__ == "__main__":
    P = {"mu": 3.99, "rho": 2.59, "k": 5.3, "lam": 0.7}
    out = []

    def say(s=""):
        print(s); out.append(s); sys.stdout.flush()

    say("== Dogrulama: AES S-box ==")
    say(HEADER)
    say(row("AES", analyze(aes_sbox())))

    say("\n== 2D-LCCM Lyapunov usleri (mu=3.99, rho=2.59, k=5.3, lam=0.7) ==")
    le = lyapunov(P, n_iter=60000)
    say(f"LE1 = {le[0]:.4f}, LE2 = {le[1]:.4f}  -> {'hiperkaotik' if le[1] > 0 else 'kaotik'}")

    say("\n== Ham (optimizasyonsuz) S-box'lar: 200 farkli anahtar ==")
    rng = np.random.default_rng(2026)
    raw = []
    for t in range(200):
        x0, y0 = rng.uniform(0.05, 0.95, 2)
        s, st = generate_sbox(P, float(x0), float(y0))
        r = analyze(s)
        raw.append((r, s, st, (float(x0), float(y0))))
    nlmins = np.array([r[0]["NL_min"] for r in raw])
    nlavgs = np.array([r[0]["NL_avg"] for r in raw])
    dus = np.array([r[0]["DU"] for r in raw])
    lps = np.array([r[0]["LP"] for r in raw])
    say(f"NL_min dagilimi: min={nlmins.min()} ort={nlmins.mean():.2f} max={nlmins.max()}")
    say(f"NL_avg dagilimi: min={nlavgs.min():.2f} ort={nlavgs.mean():.2f} max={nlavgs.max():.2f}")
    say(f"DU dagilimi: min={dus.min()} ort={dus.mean():.2f} max={dus.max()}")
    say(f"LP dagilimi: min={lps.min():.4f} ort={lps.mean():.4f} max={lps.max():.4f}")
    say(f"SAC ort (200 S-box): {np.mean([r[0]['SAC_avg'] for r in raw]):.4f}")
    say(f"BIC-SAC ort (200 S-box): {np.mean([r[0]['BIC_SAC_avg'] for r in raw]):.4f}")
    say(f"BIC-NL ort (200 S-box): {np.mean([r[0]['BIC_NL_avg'] for r in raw]):.2f}")

    raw.sort(key=lambda z: (z[0]["NL_min"], z[0]["NL_avg"], -z[0]["DU"]), reverse=True)
    best_raw = raw[0]
    say("\n" + HEADER)
    say(row("LCCM ham (en iyi)", best_raw[0]))
    say(row("LCCM ham (medyan)", raw[len(raw) // 2][0]))

    say("\n== (A) Sadece koordinat NL'si hedefli takas optimizasyonu (en iyi 5 ham S-box) ==")
    opt = []
    for r, s, st, key in raw[:5]:
        so = optimize(s, P, st, iters=6000)
        opt.append((analyze(so), so, key))
    opt.sort(key=lambda z: (z[0]["NL_min"], z[0]["NL_avg"], -z[0]["DU"], -z[0]["LP"]), reverse=True)
    say(HEADER)
    for i, (r, so, key) in enumerate(opt):
        say(row(f"LCCM optA #{i+1}", r))

    say("\n== (B) Tum bilesen NL + DU hedefli takas optimizasyonu (en iyi 5 ham S-box) ==")
    optb = []
    for r, s, st, key in raw[:5]:
        so = optimize_full(s, P, st, iters=4000)
        optb.append((analyze(so), so, key))
    optb.sort(key=lambda z: (z[0]["comp_NL"], -z[0]["DU"], z[0]["NL_min"], z[0]["NL_avg"]), reverse=True)
    say(HEADER)
    for i, (r, so, key) in enumerate(optb):
        say(row(f"LCCM optB #{i+1}", r))

    br, bs, bkey = optb[0]
    say(f"\n== Onerilen S-box (LCCM optB #1), anahtar x0={bkey[0]:.10f}, y0={bkey[1]:.10f} ==")
    say(fmt_sbox(bs))
    say(f"Koordinat NL'leri: {br['NL_all']}")
    say(f"Tum bilesenler uzerinden NL (min over b!=0): {br['comp_NL']}")
    say(f"Cebirsel dereceler: {algebraic_degree(bs)}")
    say("SAC matrisi:")
    for rr in br["SAC_matrix"]:
        say("  " + " ".join(f"{v:.4f}" for v in rr))

    say(f"\n== Ham en iyi S-box, anahtar x0={best_raw[3][0]:.10f}, y0={best_raw[3][1]:.10f} ==")
    say(fmt_sbox(best_raw[1]))

    with open("lccm_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
