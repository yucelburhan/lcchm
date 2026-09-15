"""Paper figures: English labels, 300 dpi, written to figures_paper/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

import lccm_figures as F
import lccm_sbox as L

OUT = "figures_paper"
os.makedirs(OUT, exist_ok=True)
DPI = 300
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10})
P0 = F.P0
LAB = {"mu": r"$\mu$", "rho": r"$\rho$", "k": r"$k$", "lam": r"$\lambda$", "beta": r"$\beta$"}
RANGES = {"mu": (0.0, 4.0), "rho": (0.0, 3.0), "k": (0.0, 8.0), "lam": (0.0, 1.5)}


def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (prm, (lo, hi)) in zip(axes.ravel(), RANGES.items()):
        vals = np.linspace(lo, hi, 1500)
        X = F.bifurcation(prm, vals)
        ax.plot(np.repeat(vals[None, :], X.shape[0], 0).ravel(), X.ravel(), ",k", alpha=0.25)
        ax.set_xlabel(LAB[prm]); ax.set_ylabel(r"$x_n$"); ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
    for ax, t in zip(axes.ravel(), "abcd"):
        ax.set_title(f"({t})", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/Fig1_bifurcation.png", dpi=DPI); plt.close(fig)


def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (prm, (lo, hi)) in zip(axes.ravel(), RANGES.items()):
        vals = np.linspace(lo, hi, 600)
        p = {k: np.full(len(vals), v) for k, v in P0.items()}
        p[prm] = vals
        le = F.lyap_v(p["mu"], p["rho"], p["k"], p["lam"])
        ax.plot(vals, le[:, 0], "r-", lw=0.8, label=r"$LE_1$")
        ax.plot(vals, le[:, 1], "b-", lw=0.8, label=r"$LE_2$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(LAB[prm]); ax.set_ylabel("Lyapunov exponent"); ax.set_xlim(lo, hi); ax.legend(loc="center right", fontsize=8)
    for ax, t in zip(axes.ravel(), "abcd"):
        ax.set_title(f"({t})", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/Fig2_lyapunov_1d.png", dpi=DPI); plt.close(fig)


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    n = 200
    MU, RHO = np.meshgrid(np.linspace(0, 4, n), np.linspace(0, 3, n))
    le = F.lyap_v(MU, RHO, P0["k"], P0["lam"], n_iter=1500, discard=300)[..., 0]
    im = axes[0].imshow(le, origin="lower", extent=[0, 4, 0, 3], aspect="auto", cmap="jet", vmin=0)
    axes[0].set_xlabel(r"$\mu$"); axes[0].set_ylabel(r"$\rho$"); axes[0].set_title(r"(a) $LE_1$, $k=5.3$, $\lambda=0.7$", loc="left")
    fig.colorbar(im, ax=axes[0])
    K, LAM = np.meshgrid(np.linspace(0, 8, n), np.linspace(0, 1.5, n))
    le2 = F.lyap_v(P0["mu"], P0["rho"], K, LAM, n_iter=1500, discard=300)[..., 0]
    im = axes[1].imshow(le2, origin="lower", extent=[0, 8, 0, 1.5], aspect="auto", cmap="jet", vmin=0)
    axes[1].set_xlabel(r"$k$"); axes[1].set_ylabel(r"$\lambda$"); axes[1].set_title(r"(b) $LE_1$, $\mu=3.99$, $\rho=2.59$", loc="left")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout(); fig.savefig(f"{OUT}/Fig3_lyapunov_map.png", dpi=DPI); plt.close(fig)


def fig4():
    d = np.load("scan_hyper.npz")
    planes = {"beta-lam": ("beta", (0, 0.5), "lam", (0, 2.0)), "k-lam": ("k", (0, 8), "lam", (0, 2.0)),
              "mu-lam": ("mu", (0, 4), "lam", (0, 2.0)), "rho-lam": ("rho", (0, 3), "lam", (0, 2.0)),
              "mu-rho": ("mu", (0, 4), "rho", (0, 3)), "k-beta": ("k", (0, 8), "beta", (0, 0.5))}
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.6))
    N = 160
    for j, (name, (a, (alo, ahi), b, (blo, bhi))) in enumerate(planes.items()):
        le = d[name]
        im = axes[0, j].imshow(le[..., 0], origin="lower", extent=[alo, ahi, blo, bhi], aspect="auto", cmap="jet", vmin=0)
        axes[0, j].set_title(rf"$LE_1$ ({LAB[a]}, {LAB[b]})", fontsize=9); fig.colorbar(im, ax=axes[0, j])
        arr = le[..., 1]; m = np.percentile(np.abs(arr), 99)
        im = axes[1, j].imshow(arr, origin="lower", extent=[alo, ahi, blo, bhi], aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m)
        axes[1, j].contour(np.linspace(alo, ahi, N), np.linspace(blo, bhi, N), arr, levels=[0], colors="k", linewidths=0.7)
        axes[1, j].set_title(rf"$LE_2$ ({LAB[a]}, {LAB[b]})", fontsize=9); fig.colorbar(im, ax=axes[1, j])
        for i in (0, 1):
            axes[i, j].set_xlabel(LAB[a]); axes[i, j].set_ylabel(LAB[b])
    fig.tight_layout(); fig.savefig(f"{OUT}/Fig4_hyperchaos_scan.png", dpi=DPI); plt.close(fig)


def figS1():
    N = 20000
    x, y = 0.31, 0.67; xs, ys = [], []
    for n in range(N + 500):
        x, y = L.lccm_step(x, y, n, P0)
        if n >= 500: xs.append(x); ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    axes[0, 0].plot(xs, ys, ",k", alpha=0.4); axes[0, 0].set_xlabel(r"$x_n$"); axes[0, 0].set_ylabel(r"$y_n$"); axes[0, 0].set_title("(a) phase portrait", loc="left")
    axes[0, 1].hist(xs, bins=100, color="tab:red", alpha=0.6, label=r"$x_n$", density=True)
    axes[0, 1].hist(ys, bins=100, color="tab:blue", alpha=0.6, label=r"$y_n$", density=True)
    axes[0, 1].set_title("(b) orbit histograms", loc="left"); axes[0, 1].legend(fontsize=8)
    t1, t2 = [], []; x1, y1, x2, y2 = 0.31, 0.67, 0.31 + 1e-14, 0.67
    for n in range(80):
        x1, y1 = L.lccm_step(x1, y1, n, P0); t1.append(x1); x2, y2 = L.lccm_step(x2, y2, n, P0); t2.append(x2)
    axes[1, 0].plot(t1, "r.-", lw=0.8, ms=3, label=r"$x_0=0.31$"); axes[1, 0].plot(t2, "b.-", lw=0.8, ms=3, label=r"$x_0=0.31+10^{-14}$")
    axes[1, 0].set_xlabel("n"); axes[1, 0].set_ylabel(r"$x_n$"); axes[1, 0].set_title("(c) initial-value sensitivity", loc="left"); axes[1, 0].legend(fontsize=8)
    c = np.corrcoef(xs[:-1], xs[1:])[0, 1]
    axes[1, 1].plot(xs[:-1], xs[1:], ",k", alpha=0.4); axes[1, 1].set_xlabel(r"$x_n$"); axes[1, 1].set_ylabel(r"$x_{n+1}$")
    axes[1, 1].set_title(f"(d) return map, r = {c:.4f}", loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/FigS1_trajectory.png", dpi=DPI); plt.close(fig)


def figS2():
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    n_keep, disc, M = 150, 400, 1200
    specs = [("(a) logistic", np.linspace(2.5, 4, M), lambda r, x: r * x * (1 - x), r"$\mu$"),
             ("(b) cubic", np.linspace(1.5, 3, M), lambda r, x: r * x * (1 - x * x), r"$\rho$"),
             ("(c) Chebyshev", np.linspace(0, 8, M), lambda r, x: np.cos(r * np.arccos(np.clip(x, -1, 1))), r"$k$")]
    for ax, (title, r, f, lab) in zip(axes[:3], specs):
        x = np.full(M, 0.31); pts = []
        for n in range(disc + n_keep):
            x = f(r, x)
            if n >= disc: pts.append(x.copy())
        ax.plot(np.repeat(r[None], n_keep, 0).ravel(), np.array(pts).ravel(), ",k", alpha=0.3); ax.set_title(title, loc="left"); ax.set_xlabel(lab)
    vals = np.linspace(0, 4, M); X = F.bifurcation("mu", vals, n_keep=n_keep, discard=disc)
    axes[3].plot(np.repeat(vals[None], n_keep, 0).ravel(), X.ravel(), ",k", alpha=0.3); axes[3].set_title("(d) 2D-LCChM", loc="left"); axes[3].set_xlabel(r"$\mu$")
    for ax in axes: ax.set_ylabel(r"$x_n$")
    fig.tight_layout(); fig.savefig(f"{OUT}/FigS2_seed_maps.png", dpi=DPI); plt.close(fig)


def figS3():
    import lccm_scan_beta as S
    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (prm, (lo, hi)) in zip(axes.ravel(), RANGES.items()):
        v = np.linspace(lo, hi, 300)
        for beta in betas:
            p = {kk: np.full(300, vv) for kk, vv in S.BASE.items()}; p[prm] = v
            le = S.lyap_v(p["mu"], p["rho"], p["k"], p["lam"], beta)
            ax.plot(v, le[:, 0], lw=0.9, label=rf"$\beta$={beta}")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(LAB[prm]); ax.set_ylabel(r"$LE_1$"); ax.legend(fontsize=6.5, ncol=2)
    for ax, t in zip(axes.ravel(), "abcd"):
        ax.set_title(f"({t})", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/FigS3_beta_scan.png", dpi=DPI); plt.close(fig)


def fig5_images():
    import lccm_image_enc as E, lccm_image_enc_multi as M
    names = [n for n in M.IMAGES if "256" not in n and "Siyah" not in n and "Beyaz" not in n]
    en = {"Baboon (mandril)": "Baboon", "Woman (dark hair)": "Woman (dark hair)", "Woman (blonde)": "Woman (blonde)"}
    half = (len(names) + 1) // 2
    for part, group in enumerate((names[:half], names[half:]), start=1):
        fig, ax = plt.subplots(4, len(group), figsize=(2.6 * len(group), 10))
        for j, name in enumerate(group):
            im0 = M.load(name, M.IMAGES[name]); c, _ = E.encrypt_pd(im0, M.X0, M.Y0)
            ax[0, j].imshow(im0, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(en.get(name, name), fontsize=9); ax[0, j].axis("off")
            ax[1, j].hist(im0.ravel(), bins=256, range=(0, 255), color="k"); ax[1, j].set_xlim(0, 255); ax[1, j].set_yticks([])
            ax[2, j].imshow(c, cmap="gray", vmin=0, vmax=255); ax[2, j].axis("off")
            ax[3, j].hist(c.ravel(), bins=256, range=(0, 255), color="k"); ax[3, j].set_xlim(0, 255); ax[3, j].set_yticks([])
        ax[1, 0].set_ylabel("histogram (plain)"); ax[3, 0].set_ylabel("histogram (cipher)")
        fig.tight_layout(); fig.savefig(f"{OUT}/Fig5{'ab'[part-1]}_images.png", dpi=200); plt.close(fig)
    # S4/S5: example with wrong key + correlation plots
    from matplotlib import cbook
    from PIL import Image
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        img = np.array(Image.open(f).convert("L").resize((512, 512), Image.LANCZOS), dtype=np.uint8)
    c, hh = E.encrypt_pd(img, M.X0, M.Y0); d = E.decrypt_pd(c, M.X0, M.Y0, hh); w = E.decrypt_pd(c, M.X0 + 1e-14, M.Y0, hh)
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.8))
    for j, (t, a) in enumerate((("(a) plain", img), ("(b) cipher", c), ("(c) decrypted, correct key", d), (r"(d) decrypted, $x_0+10^{-14}$", w))):
        ax[0, j].imshow(a, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(t, fontsize=9); ax[0, j].axis("off")
        ax[1, j].hist(a.ravel(), bins=256, range=(0, 255), color="k"); ax[1, j].set_xlim(0, 255); ax[1, j].set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/FigS4_encryption_example.png", dpi=200); plt.close(fig)
    g = np.random.default_rng(1); fig, ax = plt.subplots(2, 3, figsize=(9, 5.6))
    for i, (t, a) in enumerate((("plain", img), ("cipher", c))):
        af = a.astype(float); rr = g.integers(0, 511, 3000); cc = g.integers(0, 511, 3000)
        for j, (nm, (dr, dc)) in enumerate({"horizontal": (0, 1), "vertical": (1, 0), "diagonal": (1, 1)}.items()):
            ax[i, j].plot(af[rr, cc], af[rr + dr, cc + dc], ".k", ms=1.2); ax[i, j].set_title(f"{t}, {nm}", fontsize=9); ax[i, j].set_xlim(0, 255); ax[i, j].set_ylim(0, 255)
    fig.tight_layout(); fig.savefig(f"{OUT}/FigS5_correlation.png", dpi=200); plt.close(fig)


def run(name):
    globals()[name](); return name


if __name__ == "__main__":
    with Pool(6) as pool:
        for n in pool.imap_unordered(run, ["fig3", "figS3", "fig1", "fig2", "fig4", "figS1", "figS2", "fig5_images"]):
            print(n, "ok", flush=True)
