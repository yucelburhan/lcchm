"""
2D-LCChM map-analysis figures (working versions):
  fig1_bifurcation.png  : bifurcation diagrams (x_n) versus mu, rho, k, lambda
  fig2_lyapunov_1d.png  : LE1, LE2 along the same parameter ranges
  fig3_lyapunov_map.png : LE1 on the (mu,rho) and (k,lambda) planes
  fig4_trajectory.png   : phase portrait, x/y histograms, time series, initial-value sensitivity
  fig5_seed_maps.png    : bifurcations of the seed maps (logistic, cubic, Chebyshev) vs. 2D-LCChM
Vectorised NumPy; the map is checked against lccm_sbox.lccm_step.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

import lccm_sbox as L

P0 = {"mu": 3.99, "rho": 2.59, "k": 5.3, "lam": 0.7}
PHI = (math.sqrt(5.0) - 1.0) / 2.0


def cheb_v(k, u):
    return np.cos(k * np.arccos(np.clip(u, -1.0, 1.0)))


def step_v(x, y, n, mu, rho, k, lam):
    th = 2.0 * np.pi * PHI * n
    xn = (mu * x * (1 - x) + cheb_v(k, 2 * y - 1) + lam * np.sin(np.pi * (x + y) + th)) % 1.0
    yn = (rho * y * (1 - y * y) + cheb_v(k, 2 * xn - 1) + lam * np.cos(np.pi * xn * y + th)) % 1.0
    return xn, yn


def lyap_v(mu, rho, k, lam, x0=0.31, y0=0.67, n_iter=3000, discard=500):
    """Vectorised QR Lyapunov exponents: mu, rho, k, lam are arrays of equal shape."""
    mu, rho, k, lam = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in (mu, rho, k, lam)])
    shp = mu.shape
    x = np.full(shp, x0); y = np.full(shp, y0)
    for n in range(discard):
        x, y = step_v(x, y, n, mu, rho, k, lam)
    Q = np.broadcast_to(np.eye(2), shp + (2, 2)).copy()
    s = np.zeros(shp + (2,))

    def dcheb(u):
        u = np.clip(u, -1 + 1e-12, 1 - 1e-12)
        return k * np.sin(k * np.arccos(u)) / np.sqrt(1 - u * u)

    for n in range(discard, discard + n_iter):
        th = 2 * np.pi * PHI * n
        a = np.pi * (x + y) + th
        dxx = mu * (1 - 2 * x) + lam * np.pi * np.cos(a)
        dxy = 2 * dcheb(2 * y - 1) + lam * np.pi * np.cos(a)
        xn = (mu * x * (1 - x) + cheb_v(k, 2 * y - 1) + lam * np.sin(a)) % 1.0
        b = np.pi * xn * y + th
        dyxn = 2 * dcheb(2 * xn - 1) - lam * np.pi * y * np.sin(b)
        dyy = rho * (1 - 3 * y * y) - lam * np.pi * xn * np.sin(b)
        yn = (rho * y * (1 - y * y) + cheb_v(k, 2 * xn - 1) + lam * np.cos(b)) % 1.0
        J = np.empty(shp + (2, 2))
        J[..., 0, 0] = dxx; J[..., 0, 1] = dxy
        J[..., 1, 0] = dyxn * dxx; J[..., 1, 1] = dyy + dyxn * dxy
        Q, R = np.linalg.qr(J @ Q)
        s += np.log(np.abs(np.diagonal(R, axis1=-2, axis2=-1)) + 1e-300)
        x, y = xn, yn
    return s / n_iter


def bifurcation(param, values, n_keep=200, discard=500):
    p = {k: np.full(len(values), v) for k, v in P0.items()}
    p[param] = np.asarray(values, dtype=np.float64)
    x = np.full(len(values), 0.31); y = np.full(len(values), 0.67)
    xs = []
    for n in range(discard + n_keep):
        x, y = step_v(x, y, n, p["mu"], p["rho"], p["k"], p["lam"])
        if n >= discard:
            xs.append(x.copy())
    return np.array(xs)  # (n_keep, len(values))


RANGES = {"mu": (0.0, 4.0, r"$\mu$"), "rho": (0.0, 3.0, r"$\rho$"), "k": (0.0, 8.0, r"$k$"), "lam": (0.0, 1.5, r"$\lambda$")}


def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (prm, (lo, hi, lab)) in zip(axes.ravel(), RANGES.items()):
        vals = np.linspace(lo, hi, 1500)
        X = bifurcation(prm, vals)
        ax.plot(np.repeat(vals[None, :], X.shape[0], 0).ravel(), X.ravel(), ",k", alpha=0.25)
        ax.set_xlabel(lab); ax.set_ylabel(r"$x_n$"); ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
        ax.set_title(f"Bifurkasyon: {lab} (diğerleri sabit)")
    fig.tight_layout(); fig.savefig("fig1_bifurcation.png", dpi=200); plt.close(fig)


def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (prm, (lo, hi, lab)) in zip(axes.ravel(), RANGES.items()):
        vals = np.linspace(lo, hi, 600)
        p = {k: np.full(len(vals), v) for k, v in P0.items()}
        p[prm] = vals
        le = lyap_v(p["mu"], p["rho"], p["k"], p["lam"])
        ax.plot(vals, le[:, 0], "r-", lw=0.8, label=r"$LE_1$")
        ax.plot(vals, le[:, 1], "b-", lw=0.8, label=r"$LE_2$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(lab); ax.set_ylabel("Lyapunov üssü"); ax.set_xlim(lo, hi); ax.legend(loc="lower right")
        ax.set_title(f"Lyapunov üsleri: {lab}")
    fig.tight_layout(); fig.savefig("fig2_lyapunov_1d.png", dpi=200); plt.close(fig)


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = 200
    mu = np.linspace(0, 4, n); rho = np.linspace(0, 3, n)
    MU, RHO = np.meshgrid(mu, rho)
    le = lyap_v(MU, RHO, P0["k"], P0["lam"], n_iter=1500, discard=300)[..., 0]
    im = axes[0].imshow(le, origin="lower", extent=[0, 4, 0, 3], aspect="auto", cmap="jet", vmin=0)
    axes[0].set_xlabel(r"$\mu$"); axes[0].set_ylabel(r"$\rho$"); axes[0].set_title(r"$LE_1$ ($k$=5.3, $\lambda$=0.7)")
    fig.colorbar(im, ax=axes[0])
    k = np.linspace(0, 8, n); lam = np.linspace(0, 1.5, n)
    K, LAM = np.meshgrid(k, lam)
    le2 = lyap_v(P0["mu"], P0["rho"], K, LAM, n_iter=1500, discard=300)[..., 0]
    im = axes[1].imshow(le2, origin="lower", extent=[0, 8, 0, 1.5], aspect="auto", cmap="jet", vmin=0)
    axes[1].set_xlabel(r"$k$"); axes[1].set_ylabel(r"$\lambda$"); axes[1].set_title(r"$LE_1$ ($\mu$=3.99, $\rho$=2.59)")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout(); fig.savefig("fig3_lyapunov_map.png", dpi=200); plt.close(fig)
    return float(np.mean(le > 0)), float(np.mean(le2 > 0)), float(le.max()), float(le2.max())


def fig4():
    N = 20000
    x, y = 0.31, 0.67
    xs, ys = [], []
    for n in range(N + 500):
        x, y = L.lccm_step(x, y, n, P0)
        if n >= 500:
            xs.append(x); ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    x2, y2 = 0.31 + 1e-14, 0.67
    xs2 = []
    for n in range(80):
        x2, y2 = L.lccm_step(x2, y2, n, P0)
        xs2.append(x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(xs, ys, ",k", alpha=0.4); axes[0, 0].set_xlabel(r"$x_n$"); axes[0, 0].set_ylabel(r"$y_n$")
    axes[0, 0].set_title("Faz portresi (20 000 nokta)")
    axes[0, 1].hist(xs, bins=100, color="tab:red", alpha=0.6, label=r"$x_n$", density=True)
    axes[0, 1].hist(ys, bins=100, color="tab:blue", alpha=0.6, label=r"$y_n$", density=True)
    axes[0, 1].set_title("Yörünge histogramları"); axes[0, 1].legend()
    x1 = 0.31; y1 = 0.67; t1 = []
    for n in range(80):
        x1, y1 = L.lccm_step(x1, y1, n, P0); t1.append(x1)
    axes[1, 0].plot(t1, "r.-", lw=0.8, ms=3, label=r"$x_0$=0.31")
    axes[1, 0].plot(xs2, "b.-", lw=0.8, ms=3, label=r"$x_0$=0.31+$10^{-14}$")
    axes[1, 0].set_xlabel("n"); axes[1, 0].set_ylabel(r"$x_n$"); axes[1, 0].set_title("Başlangıç değeri hassasiyeti"); axes[1, 0].legend()
    c = np.corrcoef(xs[:-1], xs[1:])[0, 1]
    axes[1, 1].plot(xs[:-1], xs[1:], ",k", alpha=0.4)
    axes[1, 1].set_xlabel(r"$x_n$"); axes[1, 1].set_ylabel(r"$x_{n+1}$"); axes[1, 1].set_title(f"Otokorelasyon dağılımı (r = {c:.4f})")
    fig.tight_layout(); fig.savefig("fig4_trajectory.png", dpi=200); plt.close(fig)
    return c, float(xs.mean()), float(xs.std()), float(ys.mean()), float(ys.std())


def fig5():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    n_keep, disc, M = 150, 400, 1200
    r = np.linspace(2.5, 4, M); x = np.full(M, 0.31); pts = []
    for n in range(disc + n_keep):
        x = r * x * (1 - x)
        if n >= disc: pts.append(x.copy())
    axes[0].plot(np.repeat(r[None], n_keep, 0).ravel(), np.array(pts).ravel(), ",k", alpha=0.3)
    axes[0].set_title("Lojistik"); axes[0].set_xlabel(r"$\mu$")
    r = np.linspace(1.5, 3, M); x = np.full(M, 0.31); pts = []
    for n in range(disc + n_keep):
        x = r * x * (1 - x * x)
        if n >= disc: pts.append(x.copy())
    axes[1].plot(np.repeat(r[None], n_keep, 0).ravel(), np.array(pts).ravel(), ",k", alpha=0.3)
    axes[1].set_title("Kübik"); axes[1].set_xlabel(r"$\rho$")
    r = np.linspace(0, 8, M); x = np.full(M, 0.31); pts = []
    for n in range(disc + n_keep):
        x = np.cos(r * np.arccos(np.clip(x, -1, 1)))
        if n >= disc: pts.append(x.copy())
    axes[2].plot(np.repeat(r[None], n_keep, 0).ravel(), np.array(pts).ravel(), ",k", alpha=0.3)
    axes[2].set_title("Chebyshev"); axes[2].set_xlabel(r"$k$")
    vals = np.linspace(0, 4, M); X = bifurcation("mu", vals, n_keep=n_keep, discard=disc)
    axes[3].plot(np.repeat(vals[None], n_keep, 0).ravel(), X.ravel(), ",k", alpha=0.3)
    axes[3].set_title("2D-LCChM"); axes[3].set_xlabel(r"$\mu$")
    for ax in axes: ax.set_ylabel(r"$x_n$")
    fig.tight_layout(); fig.savefig("fig5_seed_maps.png", dpi=200); plt.close(fig)


def run(name):
    f = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}[name]
    return name, f()


if __name__ == "__main__":
    # check: vectorised map == lccm_sbox.lccm_step
    x, y = 0.31, 0.67; xv, yv = np.array([0.31]), np.array([0.67])
    for n in range(2000):
        x, y = L.lccm_step(x, y, n, P0)
        xv, yv = step_v(xv, yv, n, P0["mu"], P0["rho"], P0["k"], P0["lam"])
    assert abs(x - xv[0]) < 1e-9 and abs(y - yv[0]) < 1e-9, "harita uyusmazligi"
    le_ref = lyap_v(P0["mu"], P0["rho"], P0["k"], P0["lam"], n_iter=60000, discard=1000)
    print(f"LE (vektor kod, 60k): {le_ref[0]:.4f}, {le_ref[1]:.4f}")
    with Pool(5) as pool:
        for name, r in pool.imap_unordered(run, ["fig3", "fig1", "fig2", "fig4", "fig5"]):
            print(name, "tamam", r if r else "")
