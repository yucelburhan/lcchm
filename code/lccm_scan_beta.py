"""
Scan of the Chebyshev weight beta: dependence of LE1 on mu, rho, lambda and k for several beta.
Purpose: identify regimes in which every parameter contributes to the chaos.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
import lccm_figures as F

PHI = F.PHI


def step_v(x, y, n, mu, rho, k, lam, beta):
    th = 2.0 * np.pi * PHI * n
    xn = (mu * x * (1 - x) + beta * F.cheb_v(k, 2 * y - 1) + lam * np.sin(np.pi * (x + y) + th)) % 1.0
    yn = (rho * y * (1 - y * y) + beta * F.cheb_v(k, 2 * xn - 1) + lam * np.cos(np.pi * xn * y + th)) % 1.0
    return xn, yn


def lyap_v(mu, rho, k, lam, beta, x0=0.31, y0=0.67, n_iter=2000, discard=400):
    mu, rho, k, lam, beta = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in (mu, rho, k, lam, beta)])
    shp = mu.shape
    x = np.full(shp, x0); y = np.full(shp, y0)
    for n in range(discard):
        x, y = step_v(x, y, n, mu, rho, k, lam, beta)
    Q = np.broadcast_to(np.eye(2), shp + (2, 2)).copy()
    s = np.zeros(shp + (2,))

    def dcheb(u):
        u = np.clip(u, -1 + 1e-12, 1 - 1e-12)
        return k * np.sin(k * np.arccos(u)) / np.sqrt(1 - u * u)

    for n in range(discard, discard + n_iter):
        th = 2 * np.pi * PHI * n
        a = np.pi * (x + y) + th
        dxx = mu * (1 - 2 * x) + lam * np.pi * np.cos(a)
        dxy = 2 * beta * dcheb(2 * y - 1) + lam * np.pi * np.cos(a)
        xn = (mu * x * (1 - x) + beta * F.cheb_v(k, 2 * y - 1) + lam * np.sin(a)) % 1.0
        b = np.pi * xn * y + th
        dyxn = 2 * beta * dcheb(2 * xn - 1) - lam * np.pi * y * np.sin(b)
        dyy = rho * (1 - 3 * y * y) - lam * np.pi * xn * np.sin(b)
        yn = (rho * y * (1 - y * y) + beta * F.cheb_v(k, 2 * xn - 1) + lam * np.cos(b)) % 1.0
        J = np.empty(shp + (2, 2))
        J[..., 0, 0] = dxx; J[..., 0, 1] = dxy
        J[..., 1, 0] = dyxn * dxx; J[..., 1, 1] = dyy + dyxn * dxy
        Q, R = np.linalg.qr(J @ Q)
        s += np.log(np.abs(np.diagonal(R, axis1=-2, axis2=-1)) + 1e-300)
        x, y = xn, yn
    return s / n_iter


BASE = dict(mu=3.99, rho=2.59, k=5.3, lam=0.7)
RANGES = {"mu": (0, 4), "rho": (0, 3), "k": (0, 8), "lam": (0, 1.5)}


def job(beta):
    out = {}
    for prm, (lo, hi) in RANGES.items():
        v = np.linspace(lo, hi, 300)
        p = {kk: np.full(300, vv) for kk, vv in BASE.items()}
        p[prm] = v
        le = lyap_v(p["mu"], p["rho"], p["k"], p["lam"], beta)
        out[prm] = (v, le[:, 0], le[:, 1])
    return beta, out


if __name__ == "__main__":
    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    with Pool(len(betas)) as pool:
        res = dict(pool.map(job, betas))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, prm in zip(axes.ravel(), RANGES):
        for beta in betas:
            v, le1, le2 = res[beta][prm]
            ax.plot(v, le1, lw=1, label=f"β={beta}")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(prm); ax.set_ylabel("LE1"); ax.set_title(f"LE1 vs {prm}")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("scan_beta.png", dpi=150)
    lines = []
    for beta in betas:
        row = [f"beta={beta:<5}"]
        for prm in RANGES:
            v, le1, le2 = res[beta][prm]
            row.append(f"{prm}: LE1 {le1.min():+.2f}..{le1.max():+.2f} (>0: {100*np.mean(le1>0):.0f}%), LE2 max {le2.max():+.2f}")
        lines.append("  ".join(row))
    print("\n".join(lines))
    open("scan_beta.txt", "w", encoding="utf-8").write("\n".join(lines))
