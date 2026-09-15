"""
Multi-key C1 search and selection of the best S-box.
New keys plus the 6 keys of lccm_coord.py are evaluated together; keys are stored at full precision.

Usage: python lccm_multikey.py <n_new_keys> <iterations>
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import numpy as np
from multiprocessing import Pool

import lccm_sbox as L
import lccm_optimizers as O
import lccm_coord as C


def rank(a):
    return (a["comp_NL"], -a["DU"], -a["LP"], a["NL_min"], a["NL_avg"], a["BIC_NL_min"], a["BIC_NL_avg"],
            -abs(a["SAC_avg"] - 0.5), -abs(a["BIC_SAC_avg"] - 0.5), -a["fixed_points"])


def main():
    n = int(sys.argv[1]); iters = int(sys.argv[2])
    g_old = np.random.default_rng(2026)
    old_keys = [tuple(map(float, g_old.uniform(0.05, 0.95, 2))) for _ in range(6)]
    g_new = np.random.default_rng(7777)
    new_keys = [tuple(map(float, g_new.uniform(0.05, 0.95, 2))) for _ in range(n)]
    out = []

    def say(x=""):
        print(x); out.append(x); sys.stdout.flush()

    say(f"Cok anahtarli C1: {n} yeni anahtar (+6 onceki), {iters} iterasyon")
    entries = []
    old_final = np.load("lccm_coord_all.npy").astype(np.int64)  # in the order C1 t0..t5
    for k, s in zip(old_keys, old_final):
        entries.append(dict(src="onceki", key=k, sbox=s))
    with Pool(11) as pool:
        for r in pool.imap_unordered(C.job, [("C1", i, k, iters) for i, k in enumerate(new_keys)]):
            v, sid, key, best, f104, visits, fin, dt = r
            if fin:
                entries.append(dict(src="yeni", key=key, sbox=fin[0]))
                a = L.analyze(fin[0])
                say(f"  yeni t{sid}: cNL={a['comp_NL']} koordNL ort={a['NL_avg']:.2f} DU={a['DU']} "
                    f"LP={a['LP']:.4f} SAC={a['SAC_avg']:.4f} ({dt:.0f}s)")
            else:
                say(f"  yeni t{sid}: NL104'e ulasilamadi ({dt:.0f}s)")

    for e in entries:
        e["a"] = L.analyze(e["sbox"])
    A = [e["a"] for e in entries]
    say(f"\nToplam {len(entries)} S-box ({n} yeni anahtardan {sum(e['src'] == 'yeni' for e in entries)} tanesi NL104'e ulasti)")
    for lab, f in (("bilesen NL", "comp_NL"), ("DU", "DU"), ("koordinat NL ort", "NL_avg"), ("koordinat NL min", "NL_min"),
                   ("BIC-NL min", "BIC_NL_min")):
        vals, cnts = np.unique([a[f] for a in A], return_counts=True)
        say(f"  {lab:<17}: " + ", ".join(f"{v:g}:{c}" for v, c in zip(vals, cnts)))
    say(f"  LP: " + ", ".join(f"{v:.4f}:{c}" for v, c in zip(*np.unique([a['LP'] for a in A], return_counts=True))))

    entries.sort(key=lambda e: rank(e["a"]), reverse=True)
    say("\n== En iyi 5 ==")
    say(L.HEADER)
    for i, e in enumerate(entries[:5]):
        say(L.row(f"#{i+1} {e['src']}", e["a"]))
        say(f"    anahtar x0={e['key'][0]!r}, y0={e['key'][1]!r}")
    b = entries[0]; a = b["a"]; t = b["sbox"]
    D = O.ddt(t)
    say(f"\n== Secilen S-box ==")
    say(f"anahtar x0={b['key'][0]!r}, y0={b['key'][1]!r}")
    say(L.HEADER)
    say(L.row("AES (referans)", L.analyze(L.aes_sbox())))
    say(L.row("secilen", a))
    say(L.fmt_sbox(t))
    say(f"Koordinat NL: {a['NL_all']}")
    say(f"DDT'de 8 degerli girdi sayisi: {int((D == 8).sum())}")
    say(f"Cebirsel dereceler: {L.algebraic_degree(t)}")
    say(f"SAC min/maks: {a['SAC_min']:.4f}/{a['SAC_max']:.4f}; BIC-SAC min/maks: {a['BIC_SAC_min']:.4f}/{a['BIC_SAC_max']:.4f}")
    say("SAC matrisi:")
    for rr in a["SAC_matrix"]:
        say("  " + " ".join(f"{v:.4f}" for v in rr))
    np.save("lccm_multikey_best.npy", t)
    np.save("lccm_multikey_all.npy", np.array([e["sbox"] for e in entries]))
    json.dump([dict(src=e["src"], x0=e["key"][0], y0=e["key"][1], comp_NL=e["a"]["comp_NL"], DU=e["a"]["DU"],
                    LP=e["a"]["LP"], NL_avg=e["a"]["NL_avg"], NL_min=e["a"]["NL_min"], BIC_NL_avg=e["a"]["BIC_NL_avg"],
                    SAC_avg=e["a"]["SAC_avg"], BIC_SAC_avg=e["a"]["BIC_SAC_avg"], fixed_points=e["a"]["fixed_points"])
               for e in entries], open("lccm_multikey_keys.json", "w", encoding="utf-8"), indent=1)
    with open("lccm_multikey_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
