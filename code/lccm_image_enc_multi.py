"""
Image-encryption analysis on the standard test images (lccm_image_enc scheme, S-box A).
Colour images are converted to grey (L), 512x512. Table: entropy, correlation (H/V/D), chi2, NPCR, UACI,
key sensitivity, decryption correctness. Figure: plain/cipher images and histograms.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
import lccm_image_enc as E

# Standard test images (imageprocessingplace.com "standard test images", 512x512 grey; Lena 256 added)
IMAGES = {"Lena": "lena_gray_512.tif", "Cameraman": "cameraman.tif", "Baboon (mandril)": "mandril_gray.tif",
          "Peppers": "peppers_gray.tif", "House": "house.tif", "Jetplane": "jetplane.tif", "Lake": "lake.tif",
          "Livingroom": "livingroom.tif", "Pirate": "pirate.tif", "Walkbridge": "walkbridge.tif",
          "Woman (dark hair)": "woman_darkhair.tif", "Woman (blonde)": "woman_blonde.tif",
          "Lena 256": ("lena_gray_256.tif", 256), "Siyah (256x256)": "black", "Beyaz (256x256)": "white"}
X0, Y0 = 0.3141592653589793, 0.2718281828459045


def read_tif_raw(path):
    """Raw reader for uncompressed 8-bit TIFF with SamplesPerPixel 1 or 2 (grey [+alpha]); PIL cannot open SPP=2."""
    import struct
    d = open(path, "rb").read()
    bo = "<" if d[:2] == b"II" else ">"
    off = struct.unpack(bo + "I", d[4:8])[0]
    n = struct.unpack(bo + "H", d[off:off + 2])[0]
    tags = {}
    for i in range(n):
        e = d[off + 2 + 12 * i: off + 14 + 12 * i]
        tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
        sz = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}.get(typ)
        fmt = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d"}.get(typ)
        if sz is None or tag not in (256, 257, 258, 259, 273, 277, 278, 279):
            continue
        raw = e[8:12] if cnt * sz <= 4 else d[struct.unpack(bo + "I", e[8:12])[0]:][: cnt * sz]
        tags[tag] = struct.unpack(bo + fmt * cnt, raw[: cnt * sz])
    w, h, spp = tags[256][0], tags[257][0], tags.get(277, (1,))[0]
    assert tags.get(259, (1,))[0] == 1 and tags[258][0] == 8, "yalnizca sikistirilmamis 8-bit"
    buf = b"".join(d[o:o + c] for o, c in zip(tags[273], tags[279]))
    a = np.frombuffer(buf, dtype=np.uint8)[: w * h * spp].reshape(h, w, spp)
    return a[:, :, 0].copy()


def load(name, path):
    if path == "black":
        return np.zeros((256, 256), dtype=np.uint8)
    if path == "white":
        return np.full((256, 256), 255, dtype=np.uint8)
    if path is None:
        from matplotlib import cbook
        with cbook.get_sample_data("grace_hopper.jpg") as f:
            im = Image.open(f).convert("L").resize((512, 512), Image.LANCZOS)
        return np.array(im, dtype=np.uint8)
    size = 512
    if isinstance(path, tuple):
        path, size = path
    try:
        im = Image.open(path).convert("L")
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        return np.array(im, dtype=np.uint8)
    except Exception:
        a = read_tif_raw(path)
        assert a.shape == (size, size), (path, a.shape)
        return a


def job(item):
    name, path = item
    im0 = load(name, path)
    c, hh = E.encrypt_pd(im0, X0, Y0); d = E.decrypt_pd(c, X0, Y0, hh)
    im1 = im0.copy(); im1[im0.shape[0] // 2, im0.shape[1] // 2] ^= 1
    c1, _ = E.encrypt_pd(im1, X0, Y0)
    npcr, uaci = E.npcr_uaci(c, c1)
    cw, _ = E.encrypt_pd(im0, X0 + 1e-14, Y0)
    npcr_k, uaci_k = E.npcr_uaci(c, cw)
    dwrong = E.decrypt_pd(c, X0 + 1e-14, Y0, hh)
    cp = E.correlations(im0) if im0.std() > 0 else {"yatay": float("nan"), "dikey": float("nan"), "kosegen": float("nan")}
    cc = E.correlations(c)
    # multi-key (10 keys, random pixel) NPCR/UACI/chi2
    g = np.random.default_rng(11); mk = []
    for _ in range(10):
        x0, y0 = map(float, g.uniform(0.05, 0.95, 2)); ck, _ = E.encrypt_pd(im0, x0, y0)
        imk = im0.copy(); rr, cc2 = g.integers(0, im0.shape[0]), g.integers(0, im0.shape[1]); imk[rr, cc2] ^= 1
        nk, uk = E.npcr_uaci(ck, E.encrypt_pd(imk, x0, y0)[0]); mk.append((nk, uk, E.chi_square(ck)))
    mk = np.array(mk)
    nc, ulo, uhi = E.npcr_uaci_limits(*im0.shape)
    return name, dict(plain=im0, cipher=c, ok=np.array_equal(d, im0), ent0=E.entropy(im0), ent=E.entropy(c),
                      cp=cp, cc=cc, chi0=E.chi_square(im0), chi=E.chi_square(c), npcr=npcr, uaci=uaci,
                      npcr_k=npcr_k, uaci_k=uaci_k, ent_wrong=E.entropy(dwrong), lim=(nc, ulo, uhi),
                      mk_npcr=mk[:, 0].mean(), mk_npcr_pass=int((mk[:, 0] > nc).sum()), mk_uaci=mk[:, 1].mean(),
                      mk_uaci_pass=int(((mk[:, 1] > ulo) & (mk[:, 1] < uhi)).sum()), mk_chi=mk[:, 2].mean(),
                      mk_chi_pass=int((mk[:, 2] < 293.2).sum()))


def main():
    with Pool(len(IMAGES)) as pool:
        res = dict(pool.map(job, list(IMAGES.items())))
    out = [f"Goruntu sifreleme (Aday A S-box, 2 tur, ileri+geri yayilim, SHA-256 duz-metne bagli anahtar), anahtar x0={X0!r}, y0={Y0!r}",
           "Kabul: entropi ~8; |r|<0.02; chi2 < 293.2 (255 sd, %5); NPCR/UACI esikleri boyuta bagli (Wu vd. 2011, alpha=0.05):",
           "  512x512: NPCR > %s, UACI %s-%s;  256x256: NPCR > %s, UACI %s-%s" % tuple(
               f"{v:.4f}" for v in E.npcr_uaci_limits(512, 512) + E.npcr_uaci_limits(256, 256)),
           f"{'Goruntu':<18} {'ent duz':>7} {'ent sif':>7} {'r_Y':>8} {'r_D':>8} {'r_K':>8} {'chi2 sif':>8} {'NPCR':>8} {'UACI':>8} {'NPCR_k':>8} {'UACI_k':>8} {'coz':>4}"]
    for name in IMAGES:
        r = res[name]
        out.append(f"{name:<18} {r['ent0']:>7.4f} {r['ent']:>7.4f} {r['cc']['yatay']:>+8.4f} {r['cc']['dikey']:>+8.4f} "
                   f"{r['cc']['kosegen']:>+8.4f} {r['chi']:>8.1f} {r['npcr']:>8.4f} {r['uaci']:>8.4f} {r['npcr_k']:>8.4f} "
                   f"{r['uaci_k']:>8.4f} {'OK' if r['ok'] else 'HATA':>4}")
    out.append(f"\nCok anahtarli (10 anahtar, rastgele piksel): ortalama ve esigi gecen sayisi")
    out.append(f"{'Goruntu':<18} {'NPCR ort':>9} {'gecen':>6} {'UACI ort':>9} {'gecen':>6} {'chi2 ort':>9} {'<293.2':>6}")
    for name in IMAGES:
        r = res[name]
        out.append(f"{name:<18} {r['mk_npcr']:>9.4f} {r['mk_npcr_pass']:>4}/10 {r['mk_uaci']:>9.4f} {r['mk_uaci_pass']:>4}/10 "
                   f"{r['mk_chi']:>9.1f} {r['mk_chi_pass']:>4}/10")
    out.append("\nDuz goruntu korelasyonlari (Y/D/K):")
    for name in IMAGES:
        cp = res[name]["cp"]
        out.append(f"  {name:<18} {cp['yatay']:+.4f} / {cp['dikey']:+.4f} / {cp['kosegen']:+.4f}   chi2 duz {res[name]['chi0']:.0f}")
    nat = [n for n in IMAGES if "Siyah" not in n and "Beyaz" not in n and "256" not in n]
    half = (len(nat) + 1) // 2
    for part, names in enumerate((nat[:half], nat[half:]), start=1):
        fig, ax = plt.subplots(4, len(names), figsize=(2.8 * len(names), 11))
        for j, name in enumerate(names):
            r = res[name]
            ax[0, j].imshow(r["plain"], cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(name, fontsize=9); ax[0, j].axis("off")
            ax[1, j].hist(r["plain"].ravel(), bins=256, range=(0, 255), color="k"); ax[1, j].set_xlim(0, 255); ax[1, j].set_yticks([])
            ax[2, j].imshow(r["cipher"], cmap="gray", vmin=0, vmax=255); ax[2, j].axis("off")
            ax[3, j].hist(r["cipher"].ravel(), bins=256, range=(0, 255), color="k"); ax[3, j].set_xlim(0, 255); ax[3, j].set_yticks([])
        ax[1, 0].set_ylabel("Histogram (düz)"); ax[3, 0].set_ylabel("Histogram (şifreli)")
        fig.tight_layout(); fig.savefig(f"fig8{'ab'[part-1]}_standard_images.png", dpi=150); plt.close(fig)
    print("\n".join(out)); open("lccm_image_enc_multi_results.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
