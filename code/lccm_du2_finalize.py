"""Applies phase 3 (output-basis selection + constant XOR) to the lccm_du2 outputs and reports."""
import numpy as np
import lccm_sbox as L
import lccm_optimizers as O
import lccm_dual as Dl


def main():
    S = np.load("lccm_du2_sboxes.npy")
    meta = [ln.strip() for ln in open("lccm_du2_meta.txt", encoding="utf-8") if ln.strip()]
    out = []

    def say(x=""):
        print(x); out.append(x)

    rows = []
    say(L.HEADER)
    for s, m in zip(S, meta):
        if not m.startswith("V2"):
            continue
        tag = m.split()[1]
        rng = O.ChaosRNG(0.37, 0.59)
        t, masks, c = Dl.phase3(s.astype(np.int64), rng, trials=400)
        a0, a = L.analyze(s.astype(np.int64)), L.analyze(t)
        say(L.row(f"V2 {tag} once", a0))
        say(L.row(f"V2 {tag} son", a))
        rows.append((a, t, masks, c, m))

    def rank(z):
        a = z[0]
        return (a["comp_NL"], -a["DU"], -a["LP"], a["NL_min"], a["NL_avg"], a["BIC_NL_min"], a["BIC_NL_avg"],
                -abs(a["SAC_avg"] - 0.5), -abs(a["BIC_SAC_avg"] - 0.5), -a["fixed_points"])

    a, t, masks, c, m = max(rows, key=rank)
    D = O.ddt(t)
    say(f"\n== Onerilen S-box: {m} ==")
    say(f"Cikis tabani maskeleri: {[f'{x:08b}' for x in masks]}, cikis sabiti: {c}")
    say(L.HEADER)
    say(L.row("AES (referans)", L.analyze(L.aes_sbox())))
    say(L.row("onerilen", a))
    say(L.fmt_sbox(t))
    say(f"Koordinat NL: {a['NL_all']}")
    say(f"DDT'de 8 degerli girdi sayisi: {int((D == 8).sum())}, 6 degerli: {int((D == 6).sum())}")
    say(f"Cebirsel dereceler: {L.algebraic_degree(t)}")
    say(f"SAC min/maks: {a['SAC_min']:.4f}/{a['SAC_max']:.4f}; BIC-SAC min/maks: {a['BIC_SAC_min']:.4f}/{a['BIC_SAC_max']:.4f}")
    say("SAC matrisi:")
    for rr in a["SAC_matrix"]:
        say("  " + " ".join(f"{v:.4f}" for v in rr))
    np.save("lccm_du2_best.npy", t)
    open("lccm_du2_final.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
