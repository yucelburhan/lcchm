"""Chosen-plaintext demonstration: key streams of different plaintexts under the same master key."""
import numpy as np
import lccm_image_enc as E
import lccm_image_enc_multi as M


def main():
    X0, Y0 = M.X0, M.Y0
    lena = M.load("Lena", "lena_gray_512.tif"); black = np.zeros((512, 512), np.uint8)
    n = 512 * 512

    def stream(img):
        x1, y1, _ = E.derive_key(img, X0, Y0)
        xs, ks = E.keystream(x1, y1, n)
        return np.argsort(xs, kind="stable"), ks

    pL, kL = stream(lena); pB, kB = stream(black)
    lena1 = lena.copy(); lena1[0, 0] ^= 1; pL1, kL1 = stream(lena1)
    out = ["Secilmis-duz-metin gosterimi (ayni ana anahtar x0, y0):",
           f"  Lena vs siyah      : permutasyonda ayni konum orani {np.mean(pL == pB):.6f} (rastgele beklenti {1/n:.6f}); "
           f"anahtar akisinda ayni bayt orani {np.mean(kL == kB):.4f} (rastgele beklenti {1/256:.4f})",
           f"  Lena vs Lena+1 bit : permutasyonda ayni konum orani {np.mean(pL == pL1):.6f}; anahtar akisinda ayni bayt orani {np.mean(kL == kL1):.4f}"]
    cB, hB = E.encrypt_pd(black, X0, Y0); cL, hL = E.encrypt_pd(lena, X0, Y0)
    att = E.decrypt_pd(cL, X0, Y0, hB)
    out.append(f"  Siyah goruntunun etiketiyle (saldirganin siyah goruntuden ogrendigi anahtar akisiyla) Lena'yi cozme: "
               f"entropi {E.entropy(att):.4f}, dogru piksel orani {np.mean(att == lena):.5f} (rastgele beklenti {1/256:.5f})")
    print("\n".join(out)); open("lccm_cpa_demo.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
