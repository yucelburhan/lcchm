"""
Key sensitivity and reproducibility.
Base key: seed t0 of lccm_coord.py. The key is perturbed by small amounts; the raw S-box, the NL-104 stage
and the full C1 pipeline outputs are compared with those of the base key.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import numpy as np
from multiprocessing import Pool

import lccm_sbox as L
import lccm_optimizers as O
import lccm_guided as G
import lccm_coord as C

g = np.random.default_rng(2026)
BASE = tuple(map(float, g.uniform(0.05, 0.95, 2)))  # key t0 of lccm_coord

VARIANTS = [
    ("temel", BASE),
    ("x0+1e-14", (BASE[0] + 1e-14, BASE[1])),
    ("x0-1e-14", (BASE[0] - 1e-14, BASE[1])),
    ("y0+1e-14", (BASE[0], BASE[1] + 1e-14)),
    ("y0-1e-14", (BASE[0], BASE[1] - 1e-14)),
    ("x0+1e-10", (BASE[0] + 1e-10, BASE[1])),
]


def job(args):
    name, key = args
    t0 = time.time()
    raw, _ = L.generate_sbox(O.P, *key)
    rng = O.ChaosRNG(0.5 * key[0] + 0.23, 0.5 * key[1] + 0.17)
    nl104, _, _ = G.guided_tabu(raw, rng, 20000, target=104)
    r = C.job(("C1", 0, key, 2500))
    final = r[6][0] if r[6] else None
    return name, key, raw, nl104, final, time.time() - t0


def diff(a, b):
    pos = int((a != b).sum())
    bits = int(sum(bin(int(v)).count("1") for v in (a ^ b))) / 2048
    return pos, bits


def main():
    out = []

    def say(x=""):
        print(x); out.append(x)

    say(f"Temel anahtar: x0={BASE[0]!r}, y0={BASE[1]!r}")
    with Pool(len(VARIANTS)) as pool:
        res = {r[0]: r for r in pool.imap_unordered(job, VARIANTS)}
    base = res["temel"]
    ref = np.load("lccm_coord_best.npy").astype(np.int64)
    same = base[4] is not None and np.array_equal(base[4], ref)
    say(f"Tekrarlanabilirlik: temel anahtarla yeniden uretilen S-box, onerilen S-box ile "
        f"{'BIREBIR AYNI' if same else 'FARKLI'}")
    say("\n(fark: degisen konum sayisi / 256, bit farki orani; ideal ~255/256 ve ~0.50)")
    say(f"{'Degisiklik':<10} {'ham konum':>9} {'ham bit':>8} {'NL104 konum':>11} {'NL104 bit':>9} "
        f"{'son konum':>9} {'son bit':>8}  son S-box: cNL / koordNL ort / DU / LP")
    for name, _ in VARIANTS[1:]:
        r = res[name]
        d0 = diff(base[2], r[2]); d1 = diff(base[3], r[3])
        if r[4] is not None:
            d2 = diff(base[4], r[4]); a = L.analyze(r[4])
            tail = f"{a['comp_NL']} / {a['NL_avg']:.2f} / {a['DU']} / {a['LP']:.4f}"
        else:
            d2 = (float("nan"), float("nan")); tail = "NL104 yok"
        say(f"{name:<10} {d0[0]:>9} {d0[1]:>8.4f} {d1[0]:>11} {d1[1]:>9.4f} {d2[0]:>9} {d2[1]:>8.4f}  {tail}")
    a = L.analyze(base[4])
    say(f"{'temel':<10} {'-':>9} {'-':>8} {'-':>11} {'-':>9} {'-':>9} {'-':>8}  "
        f"{a['comp_NL']} / {a['NL_avg']:.2f} / {a['DU']} / {a['LP']:.4f}")
    say(f"\nSureler: " + ", ".join(f"{k} {v[5]:.0f}s" for k, v in res.items()))
    open("lccm_keysens_results.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
