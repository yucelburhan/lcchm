"""Extracts S-box tables from PDF text: windows of 256 distinct values in the stream of numeric tokens."""
import re, glob, os, json
import numpy as np

HEX = re.compile(r"^[0-9A-Fa-f]{2}$")


def tokens(text):
    """Numeric tokens of the text (decimal 0-255 or two-digit hex), skipping page markers."""
    toks = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("==="):
            continue
        for w in re.split(r"[\s,;|]+", s):
            if re.fullmatch(r"\d{1,3}", w) and int(w) <= 255:
                toks.append(("d", int(w), w))
            elif HEX.fullmatch(w) and not w.isdigit():
                toks.append(("h", int(w, 16), w))
            elif re.fullmatch(r"[A-Fa-f]", w):
                toks.append(("l", int(w, 16), w))  # single-letter hex label (row/column header A-F)
            else:
                toks.append(("x", None, w))
    return [(k, v) for k, v, _ in toks], [r for _, _, r in toks]


def values_of(kinds_vals, raws):
    """Extract the values; if the table contains hex tokens with letters ('9D'), all two-character tokens are read as hex."""
    if any(k == "h" for k, _ in kinds_vals):
        return [int(r, 16) if re.fullmatch(r"[0-9A-Fa-f]{2}", r) else v for (k, v), r in zip(kinds_vals, raws)]
    return [v for _, v in kinds_vals]


def find_sboxes(text):
    toks, raws = tokens(text)
    found = []
    i = 0
    while i < len(toks):
        # Layout A: 256 consecutive values (a permutation)
        if i + 256 <= len(toks):
            win = toks[i:i + 256]
            if all(k in ("d", "h") for k, _ in win):
                vals = values_of(win, raws[i:i + 256])
                if len(set(vals)) == 256:
                    found.append((i, vals)); i += 256; continue
        # Layout B: header 0..15 (or 0..9 + A..F), then 16 rows of (row label + 16 values)
        if i + 16 + 272 <= len(toks):
            hdr = toks[i:i + 16]
            hdr_dec = all(k == "d" and v == j for j, (k, v) in enumerate(hdr))
            hdr_one = all(k == "d" and v == j + 1 for j, (k, v) in enumerate(hdr))  # 1..16 (1-based)
            hdr_hex = all(k == "d" and v == j for j, (k, v) in enumerate(hdr[:10])) and \
                all(k == "l" and v == 10 + j for j, (k, v) in enumerate(hdr[10:]))
            if hdr_dec or hdr_hex or hdr_one:
                base = 1 if hdr_one else 0
                body = toks[i + 16:i + 16 + 272]
                rows = [body[r * 17:(r + 1) * 17] for r in range(16)]
                labels_ok = all(rows[r][0][0] in ("d", "l") and rows[r][0][1] == r + base for r in range(16))
                vals_ok = all(k in ("d", "h") for r in range(16) for k, _ in rows[r][1:])
                if labels_ok and vals_ok:
                    idx = [i + 16 + r * 17 + c for r in range(16) for c in range(1, 17)]
                    vals = values_of([toks[j] for j in idx], [raws[j] for j in idx])
                    if len(set(vals)) == 256:
                        found.append((i, vals)); i += 16 + 272; continue
                # Layout C: header 0..15, then 256 values without row labels
                body = toks[i + 16:i + 16 + 256]
                if all(k in ("d", "h") for k, _ in body):
                    vals = values_of(body, raws[i + 16:i + 16 + 256])
                    if len(set(vals)) == 256:
                        found.append((i, vals)); i += 16 + 256; continue
        i += 1
    return found


if __name__ == "__main__":
    out = {}
    for f in sorted(glob.glob("makaleler/*.txt")):
        text = open(f, encoding="utf-8").read()
        sb = find_sboxes(text)
        name = os.path.basename(f)[:-4]
        print(f"{name:<40} {len(sb)} S-box bulundu")
        for k, (pos, vals) in enumerate(sb):
            out[f"{name}#{k+1}"] = vals
    json.dump(out, open("lit_sboxes_raw.json", "w"), indent=0)
    print(f"toplam {len(out)} tablo -> lit_sboxes_raw.json")
