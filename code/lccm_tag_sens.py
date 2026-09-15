"""Tag sensitivity: each of the 256 bits of the SHA-256 tag is flipped in turn and the decrypted image is examined."""
import numpy as np
from multiprocessing import Pool
import lccm_image_enc as E
import lccm_image_enc_multi as M

X0, Y0 = M.X0, M.Y0
_STATE = {}


def _init():
    if not _STATE:
        img = M.load("Lena 256", ("lena_gray_256.tif", 256))
        c, h = E.encrypt_pd(img, X0, Y0)
        _STATE.update(img=img, c=c, h=h)
    return _STATE


def job(bit):
    st = _init()
    hb = bytearray(bytes.fromhex(st["h"])); hb[bit // 8] ^= 1 << (bit % 8)
    d = E.decrypt_pd(st["c"], X0, Y0, hb.hex())
    return bit, E.entropy(d), float(np.mean(d == st["img"])), float(E.npcr_uaci(d, st["img"])[0])


def main():
    st = _init()
    assert np.array_equal(E.decrypt_pd(st["c"], X0, Y0, st["h"]), st["img"])
    with Pool(11) as pool:
        res = sorted(pool.map(job, range(256)))
    ent = np.array([r[1] for r in res]); same = np.array([r[2] for r in res]); npcr = np.array([r[3] for r in res])
    lines = ["Etiket hassasiyeti (Lena 256x256, 256 bitin her biri ayri ayri bozuldu):",
             f"  cozulen goruntu entropisi: min {ent.min():.4f} ort {ent.mean():.4f} maks {ent.max():.4f} (duz: {E.entropy(st['img']):.4f})",
             f"  dogru piksel orani: min {same.min():.5f} ort {same.mean():.5f} maks {same.max():.5f} (rastgele beklenti 0.00391)",
             f"  cozulen vs duz NPCR: min {npcr.min():.4f} ort {npcr.mean():.4f}",
             f"  cozmeyi basaran bit sayisi (dogru piksel > %1): {int((same > 0.01).sum())}/256"]
    print("\n".join(lines)); open("lccm_tag_sens.txt", "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
