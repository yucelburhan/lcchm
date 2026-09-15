"""Random Excursions / Variant: every state is evaluated as a separate test, as in NIST."""
import numpy as np
from math import sqrt
from multiprocessing import Pool
import lccm_nist as N


def job(seed):
    b = N.gen_bits(seed)
    return N.random_excursions(b)


if __name__ == "__main__":
    with Pool(11) as pool:
        res = [r for r in pool.map(job, range(100)) if r[0] is not None]
    n = len(res)
    thr = 0.99 - 3 * sqrt(0.01 * 0.99 / n)
    lines = [f"Random Excursions (uygulanabilir dizi: {n}/100, oran esigi {thr:.4f})",
             f"{'Test':<28} {'gecen':>6} {'oran':>6} {'p-dagilim':>10} {'sonuc':>6}"]
    allok = True
    for label, idx_list, key in (("RandomExcursions", range(8), 0), ("RandExcVariant", range(18), 1)):
        states = [-4, -3, -2, -1, 1, 2, 3, 4] if key == 0 else list(range(-9, 0)) + list(range(1, 10))
        for i in idx_list:
            ps = np.array([r[key][i] for r in res])
            npass = (ps >= 0.01).sum()
            h = np.histogram(ps, bins=10, range=(0, 1))[0]
            pu = N.igamc(4.5, ((h - n / 10) ** 2 / (n / 10)).sum() / 2)
            ok = npass / n >= thr and pu >= 1e-4
            allok &= ok
            lines.append(f"{label + ' x=' + str(states[i]):<28} {npass:>3}/{n:<3} {npass/n:>6.3f} {pu:>10.4f} {'GECTI' if ok else 'KALDI':>6}")
    lines.append(f"\nGenel: {'TUM DURUMLAR GECTI' if allok else 'BAZI DURUMLAR KALDI'}")
    print("\n".join(lines)); open("lccm_nist_re_results.txt", "w", encoding="utf-8").write("\n".join(lines))
