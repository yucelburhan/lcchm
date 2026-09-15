"""Hyperchaotic-region scan: LE2 > 0 on the (beta,lam), (k,lam), (mu,lam), (rho,lam), (mu,rho), (k,beta) planes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
import lccm_scan_beta as S

BASE = dict(mu=3.99, rho=2.59, k=5.3, lam=0.7, beta=0.05)
N = 160
PLANES = {
    "beta-lam": ("beta", (0, 0.5), "lam", (0, 2.0)),
    "k-lam": ("k", (0, 8), "lam", (0, 2.0)),
    "mu-lam": ("mu", (0, 4), "lam", (0, 2.0)),
    "rho-lam": ("rho", (0, 3), "lam", (0, 2.0)),
    "mu-rho": ("mu", (0, 4), "rho", (0, 3)),
    "k-beta": ("k", (0, 8), "beta", (0, 0.5)),
}


def job(name):
    a, (alo, ahi), b, (blo, bhi) = PLANES[name]
    A, B = np.meshgrid(np.linspace(alo, ahi, N), np.linspace(blo, bhi, N))
    p = {kk: np.full(A.shape, vv) for kk, vv in BASE.items()}
    p[a] = A; p[b] = B
    le = S.lyap_v(p["mu"], p["rho"], p["k"], p["lam"], p["beta"], n_iter=1500, discard=300)
    return name, le


if __name__ == "__main__":
    with Pool(len(PLANES)) as pool:
        res = dict(pool.map(job, list(PLANES)))
    fig, axes = plt.subplots(2, 6, figsize=(24, 7.5))
    lines = []
    for j, name in enumerate(PLANES):
        a, (alo, ahi), b, (blo, bhi) = PLANES[name]
        le = res[name]
        for i, (lab, arr, vmin) in enumerate((("LE1", le[..., 0], 0.0), ("LE2", le[..., 1], None))):
            ax = axes[i, j]
            if lab == "LE2":
                m = np.abs(arr).max()
                im = ax.imshow(arr, origin="lower", extent=[alo, ahi, blo, bhi], aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m)
                ax.contour(np.linspace(alo, ahi, N), np.linspace(blo, bhi, N), arr, levels=[0], colors="k", linewidths=0.8)
            else:
                im = ax.imshow(arr, origin="lower", extent=[alo, ahi, blo, bhi], aspect="auto", cmap="jet", vmin=0)
            ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{lab} ({name})"); fig.colorbar(im, ax=ax)
        hyper = (le[..., 0] > 0) & (le[..., 1] > 0)
        lines.append(f"{name:<9}: LE1>0 {100*np.mean(le[...,0]>0):5.1f}%  hiperkaotik {100*np.mean(hyper):5.1f}%  "
                     f"LE2 maks {le[...,1].max():+.3f}  (hiper bolgede LE1 ort {le[...,0][hyper].mean() if hyper.any() else float('nan'):.2f})")
    fig.tight_layout(); fig.savefig("scan_hyper.png", dpi=130)
    print("\n".join(lines))
    open("scan_hyper.txt", "w", encoding="utf-8").write("\n".join(lines))
    np.savez("scan_hyper.npz", **{k: v for k, v in res.items()})
