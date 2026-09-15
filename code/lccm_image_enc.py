"""
Grey-scale image encryption with the 2D-LCChM S-box and its security analysis.

Scheme (2 rounds; master key (x0, y0), plaintext-dependent key derivation via SHA-256):
  1. permutation : pixel permutation by the argsort of a 2D-LCChM sequence
  2. substitution: S-box A
  3. diffusion   : c_i = S[ p_i XOR k_i XOR c_{i-1} ] forward and backward (k_i: 8-bit key stream from the map)
Decryption reverses the steps.

Metrics: entropy, adjacent-pixel correlation (H/V/D), NPCR, UACI, histogram chi-square,
key sensitivity (decryption with x0 + 1e-14), decryption correctness.
Test image: matplotlib sample grace_hopper.jpg (512x512 grey) + an all-black image.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cbook
from PIL import Image

import lccm_sbox as L

P = {"mu": 3.99, "rho": 2.59, "k": 5.3, "lam": 0.7}
SBOX = np.load("lccm_multikey_best.npy").astype(np.int64)
INV = np.empty(256, dtype=np.int64); INV[SBOX] = np.arange(256)


def keystream(x0, y0, n, discard=1000):
    x, y = x0, y0
    it = 0
    for _ in range(discard):
        x, y = L.lccm_step(x, y, it, P); it += 1
    xs = np.empty(n); ks = np.empty(n, dtype=np.int64)
    for i in range(n):
        x, y = L.lccm_step(x, y, it, P); it += 1
        xs[i] = x
        ks[i] = (int(x * 1e14) ^ int(y * 1e14)) & 0xFF
    return xs, ks


def encrypt(img, x0, y0, rounds=2):
    flat = img.astype(np.int64).ravel()
    n = flat.size
    c = flat.copy()
    for r in range(rounds):
        xs, ks = keystream(x0 + r * 1e-3, y0, n)
        perm = np.argsort(xs, kind="stable")
        c = c[perm]                       # permutation
        c = SBOX[c]                       # substitution
        out = np.empty_like(c); prev = 0  # forward diffusion
        for i in range(n):
            prev = SBOX[c[i] ^ ks[i] ^ prev]
            out[i] = prev
        c = out
        out = np.empty_like(c); prev = 0  # backward diffusion (full avalanche within one round)
        for i in range(n - 1, -1, -1):
            prev = SBOX[c[i] ^ ks[n - 1 - i] ^ prev]
            out[i] = prev
        c = out
    return c.reshape(img.shape).astype(np.uint8)


def decrypt(cimg, x0, y0, rounds=2):
    c = cimg.astype(np.int64).ravel()
    n = c.size
    for r in reversed(range(rounds)):
        xs, ks = keystream(x0 + r * 1e-3, y0, n)
        perm = np.argsort(xs, kind="stable")
        p = np.empty_like(c); prev = 0     # inverse of the backward diffusion
        for i in range(n - 1, -1, -1):
            p[i] = INV[c[i]] ^ ks[n - 1 - i] ^ prev
            prev = c[i]
        c = p
        p = np.empty_like(c); prev = 0     # inverse of the forward diffusion
        for i in range(n):
            p[i] = INV[c[i]] ^ ks[i] ^ prev
            prev = c[i]
        p = INV[p]
        inv = np.empty_like(perm); inv[perm] = np.arange(n)
        c = p[inv]
    return c.reshape(cimg.shape).astype(np.uint8)


def derive_key(img, x0, y0):
    """Plaintext-dependent key: h = SHA-256(image bytes); (x0, y0) is perturbed by h.
    h is transmitted to the receiver together with the cipher image (256-bit tag)."""
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(img, dtype=np.uint8).tobytes()).digest()
    x1, y1 = keys_from_hash(x0, y0, h.hex())
    return x1, y1, h.hex()


def keys_from_hash(x0, y0, hexhash):
    """All 256 tag bits are used: each 128-bit half is reduced modulo a 53-bit modulus to give (dx, dy), so that
    every tag bit produces a change representable in double precision."""
    h = bytes.fromhex(hexhash)
    Pm = 2 ** 53 - 111
    a = int.from_bytes(h[0:16], "big") % Pm
    b = int.from_bytes(h[16:32], "big") % Pm
    dx, dy = a / Pm, b / Pm
    x1 = (x0 + dx) % 1.0; y1 = (y0 + dy) % 1.0
    x1 = x1 if 1e-9 < x1 < 1 - 1e-9 else 0.5 + dx / 4
    y1 = y1 if 1e-9 < y1 < 1 - 1e-9 else 0.5 + dy / 4
    return x1, y1


def encrypt_pd(img, x0, y0, rounds=2):
    """Plaintext-dependent encryption: returns (cipher image, SHA-256 tag)."""
    x1, y1, hh = derive_key(img, x0, y0)
    return encrypt(img, x1, y1, rounds), hh


def decrypt_pd(cimg, x0, y0, hexhash, rounds=2):
    x1, y1 = keys_from_hash(x0, y0, hexhash)
    return decrypt(cimg, x1, y1, rounds)


def entropy(img):
    h = np.bincount(img.ravel(), minlength=256) / img.size
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def correlations(img, n=5000, seed=0):
    g = np.random.default_rng(seed)
    H, W = img.shape
    r = g.integers(0, H - 1, n); c = g.integers(0, W - 1, n)
    a = img.astype(np.float64)
    out = {}
    for name, (dr, dc) in {"yatay": (0, 1), "dikey": (1, 0), "kosegen": (1, 1)}.items():
        out[name] = float(np.corrcoef(a[r, c], a[r + dr, c + dc])[0, 1])
    return out


def npcr_uaci(c1, c2):
    d = c1 != c2
    npcr = 100.0 * d.mean()
    uaci = 100.0 * (np.abs(c1.astype(np.float64) - c2.astype(np.float64)) / 255.0).mean()
    return float(npcr), float(uaci)


def npcr_uaci_limits(h, w, alpha=0.05, F=255):
    """Wu, Noonan, Agaian (2011): size-dependent NPCR critical value and UACI acceptance interval (percent)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(alpha); z2 = NormalDist().inv_cdf(alpha / 2)
    mn = h * w
    n_crit = (F - (-z) * math.sqrt(F / mn)) / (F + 1) * 100
    mu = (F + 2) / (3 * F + 3)
    sig = math.sqrt((F + 2) * (F * F + 2 * F + 3) / (18 * (F + 1) ** 2 * F * mn))
    return n_crit, (mu + z2 * sig) * 100, (mu - z2 * sig) * 100


def chi_square(img):
    h = np.bincount(img.ravel(), minlength=256)
    e = img.size / 256
    return float(((h - e) ** 2 / e).sum())


def main():
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        im = Image.open(f).convert("L").resize((512, 512), Image.LANCZOS)
    img = np.array(im, dtype=np.uint8)
    black = np.zeros((256, 256), dtype=np.uint8)
    x0, y0 = 0.3141592653589793, 0.2718281828459045
    out = []

    def say(s=""):
        print(s); out.append(s)

    say("2D-LCChM S-box (Aday A) ile goruntu sifreleme (duz-metne bagli anahtar, ileri+geri yayilim); anahtar x0=%r, y0=%r" % (x0, y0))
    results = {}
    for name, im0 in (("grace_hopper 512x512", img), ("siyah 256x256", black)):
        c, hh = encrypt_pd(im0, x0, y0)
        d = decrypt_pd(c, x0, y0, hh)
        ok = np.array_equal(d, im0)
        # NPCR/UACI: one-pixel change
        im1 = im0.copy(); im1[im0.shape[0] // 2, im0.shape[1] // 2] ^= 1
        c1, _ = encrypt_pd(im1, x0, y0)
        npcr, uaci = npcr_uaci(c, c1)
        # key sensitivity
        dwrong = decrypt_pd(c, x0 + 1e-14, y0, hh)
        cw, _ = encrypt_pd(im0, x0 + 1e-14, y0)
        npcr_k, uaci_k = npcr_uaci(c, cw)
        # tag sensitivity: decryption with one flipped tag bit
        ents = []
        for pos in (0, 15, 31):  # flip one bit at the start, middle and end of the tag
            hb = bytearray(bytes.fromhex(hh)); hb[pos] ^= 1
            ents.append(entropy(decrypt_pd(c, x0, y0, hb.hex())))
        say(f"  [{name}] SHA-256 etiketi: {hh[:16]}...; etiketin 0./15./31. baytinda 1 bit bozukken cozulen goruntu entropisi "
            + " / ".join(f"{e:.4f}" for e in ents))
        results[name] = dict(plain=im0, cipher=c, dec=d, wrong=dwrong)
        say(f"\n== {name} ==")
        say(f"  cozme dogru: {ok}")
        say(f"  entropi: duz {entropy(im0):.4f}  sifreli {entropy(c):.4f}  (ideal 8)")
        cp, cc = correlations(im0), correlations(c)
        for k in cp:
            say(f"  korelasyon {k:<8}: duz {cp[k]:+.4f}  sifreli {cc[k]:+.4f}")
        say(f"  ki-kare: duz {chi_square(im0):.1f}  sifreli {chi_square(c):.1f}  (255 sd, %5 esik 293.2)")
        nc, ulo, uhi = npcr_uaci_limits(*im0.shape)
        say(f"  NPCR: {npcr:.4f} %  (kritik %{nc:.4f})   UACI: {uaci:.4f} %  (kabul araligi {ulo:.4f}-{uhi:.4f})")
        say(f"  anahtar hassasiyeti (x0+1e-14): sifreli goruntuler arasi NPCR {npcr_k:.4f} %, UACI {uaci_k:.4f} %; "
            f"yanlis anahtarla cozulen goruntu entropisi {entropy(dwrong):.4f}")

    r = results["grace_hopper 512x512"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.2))
    for j, (t, a) in enumerate((("Düz görüntü", r["plain"]), ("Şifreli", r["cipher"]),
                                ("Çözülmüş (doğru anahtar)", r["dec"]), ("Çözülmüş (x₀+1e-14)", r["wrong"]))):
        ax[0, j].imshow(a, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(t); ax[0, j].axis("off")
        ax[1, j].hist(a.ravel(), bins=256, range=(0, 255), color="k"); ax[1, j].set_xlim(0, 255)
        ax[1, j].set_title("Histogram")
    fig.tight_layout(); fig.savefig("fig6_image_encryption.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
    g = np.random.default_rng(1)
    for i, (t, a) in enumerate((("düz", r["plain"]), ("şifreli", r["cipher"]))):
        af = a.astype(np.float64)
        rr = g.integers(0, 511, 3000); cc = g.integers(0, 511, 3000)
        for j, (nm, (dr, dc)) in enumerate({"yatay": (0, 1), "dikey": (1, 0), "köşegen": (1, 1)}.items()):
            ax[i, j].plot(af[rr, cc], af[rr + dr, cc + dc], ".k", ms=1.5)
            ax[i, j].set_title(f"{t} – {nm}"); ax[i, j].set_xlim(0, 255); ax[i, j].set_ylim(0, 255)
    fig.tight_layout(); fig.savefig("fig7_correlation.png", dpi=170); plt.close(fig)
    open("lccm_image_enc_results.txt", "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
