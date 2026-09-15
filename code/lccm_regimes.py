"""Comparison of candidate parameter regimes: LE (60k iterations), uniformity, correlation, raw S-box quality."""
import numpy as np
from multiprocessing import Pool
import lccm_scan_beta as S
import lccm_sbox as L

REGIMES = {
    "V1 (mevcut)":      dict(mu=3.99, rho=2.59, k=5.3, lam=0.7, beta=1.0),
    "M1 dengeli":       dict(mu=3.99, rho=2.59, k=5.3, lam=0.7, beta=0.2),
    "H1 hiper":         dict(mu=3.99, rho=2.59, k=5.3, lam=0.3, beta=0.05),
    "H2 hiper":         dict(mu=3.99, rho=2.59, k=5.3, lam=0.4, beta=0.1),
    "H3 hiper":         dict(mu=3.99, rho=2.59, k=5.3, lam=0.2, beta=0.05),
    "H4 hiper k=3":     dict(mu=3.99, rho=2.59, k=3.0, lam=0.3, beta=0.05),
}


def job(item):
    name, p = item
    le = S.lyap_v(p["mu"], p["rho"], p["k"], p["lam"], p["beta"], n_iter=60000, discard=1000)
    x, y = 0.31, 0.67
    xs, ys = [], []
    for n in range(101000):
        x, y = L.lccm_step(x, y, n, p)
        if n >= 1000:
            xs.append(x); ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    hx = np.bincount((xs * 64).astype(int), minlength=64)[:64]; hy = np.bincount((ys * 64).astype(int), minlength=64)[:64]
    e = len(xs) / 64
    chi = (((hx - e) ** 2 / e).sum(), ((hy - e) ** 2 / e).sum())  # 63 d.o.f., 5% threshold 82.5
    r1 = np.corrcoef(xs[:-1], xs[1:])[0, 1]
    rxy = np.corrcoef(xs, ys)[0, 1]
    # raw S-box quality (20 keys)
    g = np.random.default_rng(5)
    cnl, nlavg, du = [], [], []
    for _ in range(20):
        s, _ = L.generate_sbox(p, *map(float, g.uniform(0.05, 0.95, 2)))
        a = L.analyze(s); cnl.append(a["comp_NL"]); nlavg.append(a["NL_avg"]); du.append(a["DU"])
    # parameter sensitivity: change of LE1 for a 25% reduction of each parameter
    sens = {}
    for kk in ("mu", "rho", "k", "lam", "beta"):
        q = dict(p); q[kk] = p[kk] * 0.75
        le2 = S.lyap_v(q["mu"], q["rho"], q["k"], q["lam"], q["beta"], n_iter=20000, discard=1000)
        sens[kk] = float(le2[0] - le[0])
    return name, dict(le=le, chi=chi, r1=r1, rxy=rxy, cnl=np.mean(cnl), nlavg=np.mean(nlavg), du=np.mean(du), sens=sens,
                      xm=xs.mean(), xs=xs.std())


if __name__ == "__main__":
    with Pool(len(REGIMES)) as pool:
        res = dict(pool.map(job, list(REGIMES.items())))
    lines = [f"{'rejim':<16} {'LE1':>7} {'LE2':>7} {'chi2 x':>8} {'chi2 y':>8} {'r(x,x+1)':>9} {'r(x,y)':>7} "
             f"{'x ort':>6} {'x std':>6} {'ham cNL':>8} {'ham NLavg':>9} {'ham DU':>7}   LE1 degisimi (%25 azalma): mu rho k lam beta"]
    for name, p in REGIMES.items():
        r = res[name]
        lines.append(f"{name:<16} {r['le'][0]:>7.3f} {r['le'][1]:>7.3f} {r['chi'][0]:>8.1f} {r['chi'][1]:>8.1f} {r['r1']:>9.4f} "
                     f"{r['rxy']:>7.4f} {r['xm']:>6.3f} {r['xs']:>6.3f} {r['cnl']:>8.1f} {r['nlavg']:>9.2f} {r['du']:>7.1f}   "
                     + " ".join(f"{r['sens'][kk]:+.2f}" for kk in ("mu", "rho", "k", "lam", "beta")))
    print("\n".join(lines)); open("lccm_regimes.txt", "w", encoding="utf-8").write("\n".join(lines))
