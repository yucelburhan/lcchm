"""
Chaos-performance comparison: 2D-LCChM (working regime) vs. its seed maps and common 2D maps.
Metrics: Lyapunov exponent (mean log|f'| for 1D maps, QR method for 2D maps), permutation entropy (PE, m=5),
sample entropy (SampEn, m=2, r=0.2*std), 0-1 test (K, median), orbit uniformity (chi2, 64 bins, 100k samples),
lag-1 autocorrelation.
"""
import numpy as np
from multiprocessing import Pool
import lccm_sbox as L
import lccm_scan_beta as S
from itertools import permutations

P = L.load_params({"mu": 3.99, "rho": 2.59, "k": 5.3, "lam": 0.7, "beta": 1.0})
N = 100_000


def traj_lccm():
    x, y = 0.31, 0.67; xs = []
    for n in range(N + 1000):
        x, y = L.lccm_step(x, y, n, P)
        if n >= 1000: xs.append(x)
    return np.array(xs), None


def traj_1d(f, df, x0=0.31, to01=None):
    x = x0; xs = []; le = 0.0
    for n in range(N + 1000):
        if n >= 1000:
            le += np.log(abs(df(x)) + 1e-300); xs.append(x)
        x = f(x)
    xs = np.array(xs)
    if to01: xs = to01(xs)
    return xs, le / N


def traj_2d(f, x0=0.31, y0=0.67):
    x, y = x0, y0; xs = []
    for n in range(N + 1000):
        x, y = f(x, y)
        if n >= 1000: xs.append(x)
    return np.array(xs), None


def perm_entropy(x, m=5, tau=1):
    n = len(x) - (m - 1) * tau
    idx = np.argsort(np.stack([x[i * tau:i * tau + n] for i in range(m)], axis=1), axis=1)
    codes = (idx * (m ** np.arange(m))).sum(axis=1)
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / np.log(np.math.factorial(m) if hasattr(np, "math") else __import__("math").factorial(m)))


def sample_entropy(x, m=2, r=None, n=4000):
    x = x[:n]; r = 0.2 * x.std() if r is None else r
    def count(mm):
        emb = np.stack([x[i:n - mm + 1 + i] for i in range(mm)], axis=1)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        return ((d <= r).sum() - len(emb)) / 2
    B, A = count(m), count(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else float("inf")


def test01(x, n=5000, nc=100):
    x = x[:n]; g = np.random.default_rng(0); Ks = []
    for c in g.uniform(np.pi / 5, 4 * np.pi / 5, nc):
        p = np.cumsum(x * np.cos(np.arange(1, n + 1) * c)); q = np.cumsum(x * np.sin(np.arange(1, n + 1) * c))
        ncut = n // 10
        M = np.array([np.mean((p[j:] - p[:-j]) ** 2 + (q[j:] - q[:-j]) ** 2) for j in range(1, ncut + 1)])
        D = M - (x.mean() ** 2) * (1 - np.cos(np.arange(1, ncut + 1) * c)) / (1 - np.cos(c))
        Ks.append(np.corrcoef(np.arange(1, ncut + 1), D)[0, 1])
    return float(np.median(Ks))


def uniform_chi2(x):
    h = np.bincount(np.minimum((x * 64).astype(int), 63), minlength=64); e = len(x) / 64
    return float(((h - e) ** 2 / e).sum())


MAPS = {
    "2D-LCCM (V1)": lambda: traj_lccm(),
    "Lojistik (μ=3.99)": lambda: traj_1d(lambda x: 3.99 * x * (1 - x), lambda x: 3.99 * (1 - 2 * x)),
    "Kübik (ρ=2.59)": lambda: traj_1d(lambda x: 2.59 * x * (1 - x * x), lambda x: 2.59 * (1 - 3 * x * x), x0=0.31,
                                     to01=lambda v: (v + 1.2) / 2.4),
    "Chebyshev (k=5.3)": lambda: traj_1d(lambda x: np.cos(5.3 * np.arccos(np.clip(x, -1, 1))),
                                        lambda x: 5.3 * np.sin(5.3 * np.arccos(np.clip(x, -1 + 1e-12, 1 - 1e-12))) / np.sqrt(1 - min(x * x, 1 - 1e-12)),
                                        x0=0.31, to01=lambda v: (v + 1) / 2),
    "Sinüs (a=0.99)": lambda: traj_1d(lambda x: 0.99 * np.sin(np.pi * x), lambda x: 0.99 * np.pi * np.cos(np.pi * x)),
    "2D-LSCM (θ=0.9)": lambda: traj_2d(lambda x, y: (np.sin(np.pi * (4 * 0.9 * x * (1 - x) + (1 - 0.9) * np.sin(np.pi * y))),
                                                    np.sin(np.pi * (4 * 0.9 * y * (1 - y) + (1 - 0.9) * np.sin(np.pi * (np.sin(np.pi * (4 * 0.9 * x * (1 - x) + (1 - 0.9) * np.sin(np.pi * y))))))))),
    "Hénon (1.4, 0.3)": lambda: traj_2d(lambda x, y: (1 - 1.4 * x * x + y, 0.3 * x), x0=0.1, y0=0.1),
}


def le_2d_numeric(name):
    """Numerical Lyapunov exponents of a 2D map (finite-difference Jacobian, QR)."""
    if name == "2D-LCCM (V1)":
        le = S.lyap_v(P["mu"], P["rho"], P["k"], P["lam"], P["beta"], n_iter=60000, discard=1000)
        return float(le[0]), float(le[1])
    f = {"2D-LSCM (θ=0.9)": lambda x, y: (np.sin(np.pi * (4 * 0.9 * x * (1 - x) + 0.1 * np.sin(np.pi * y))),
                                          np.sin(np.pi * (4 * 0.9 * y * (1 - y) + 0.1 * np.sin(np.pi * np.sin(np.pi * (4 * 0.9 * x * (1 - x) + 0.1 * np.sin(np.pi * y))))))),
         "Hénon (1.4, 0.3)": lambda x, y: (1 - 1.4 * x * x + y, 0.3 * x)}[name]
    x, y = (0.31, 0.67) if "LSCM" in name else (0.1, 0.1)
    for _ in range(1000): x, y = f(x, y)
    Q = np.eye(2); s = np.zeros(2); h = 1e-7
    for _ in range(60000):
        fx, fy = f(x, y)
        J = np.array([[(f(x + h, y)[0] - fx) / h, (f(x, y + h)[0] - fx) / h],
                      [(f(x + h, y)[1] - fy) / h, (f(x, y + h)[1] - fy) / h]])
        Q, R = np.linalg.qr(J @ Q); s += np.log(np.abs(np.diag(R)) + 1e-300); x, y = fx, fy
    return float(s[0] / 60000), float(s[1] / 60000)


def job(name):
    xs, le = MAPS[name]()
    if le is None:
        le1, le2 = le_2d_numeric(name)
    else:
        le1, le2 = le, None
    if "Hénon" in name:  # rescale to [0,1] (for the uniformity statistic)
        xu = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    else:
        xu = np.clip(xs, 0, 1 - 1e-12)
    return name, dict(le1=le1, le2=le2, pe=perm_entropy(xs), se=sample_entropy(xs), k01=test01(xs),
                      chi=uniform_chi2(xu), r1=float(np.corrcoef(xs[:-1], xs[1:])[0, 1]))


if __name__ == "__main__":
    with Pool(len(MAPS)) as pool:
        res = dict(pool.map(job, list(MAPS)))
    lines = [f"{'Harita':<20} {'LE1':>7} {'LE2':>7} {'PE(m=5)':>8} {'SampEn':>7} {'0-1 K':>6} {'chi2(64)':>9} {'r(x,x+1)':>9}",
             "(PE ideal 1; SampEn buyuk iyi; K~1 kaotik; chi2 duzgunluk, 63 sd, %5 esik 82.5; 100k ornek)"]
    for name in MAPS:
        r = res[name]
        lines.append(f"{name:<20} {r['le1']:>7.3f} {(f'{r['le2']:7.3f}' if r['le2'] is not None else '      –')} "
                     f"{r['pe']:>8.4f} {r['se']:>7.3f} {r['k01']:>6.3f} {r['chi']:>9.1f} {r['r1']:>9.4f}")
    print("\n".join(lines)); open("lccm_chaos_metrics.txt", "w", encoding="utf-8").write("\n".join(lines))
