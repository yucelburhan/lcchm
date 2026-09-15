# 2D-LCChM S-box: key-dependent 8-bit S-boxes with component nonlinearity 104 from a 2D Logistic–Cubic–Chebyshev chaotic map

Code, S-boxes, results and figures accompanying the paper

> Key-dependent 8-bit S-boxes with component nonlinearity 104: Walsh-guided design over a 2D Logistic–Cubic–Chebyshev chaotic map and a re-evaluation of chaos-based S-boxes, Yücel BÜRHAN, submitted to *Chaos, Solitons & Fractals*, 2026.

Everything reported in the paper can be regenerated from this repository. All security metrics were
cross-checked by an independent brute-force implementation (`code/lccm_verify.py`) that shares no code
with the fast analyzer, and validated on the AES S-box.

## Highlights

| Property | Proposed S-box A | AES |
|---|---|---|
| Component nonlinearity (min over all 255 component functions) | **104** | 112 |
| Coordinate nonlinearity (min / avg / max over 8 output bits) | 108 / **109.50** / 110 | 112 |
| Linear probability LP | **0.09375** | 0.0625 |
| Differential uniformity DU (DP) | **8** (0.03125) | 4 |
| SAC (avg) | 0.5054 | 0.5049 |
| BIC-NL (avg) / BIC-SAC (avg) | 105.71 / 0.5036 | 112 / 0.5046 |
| Fixed points / algebraic degree | 0 / 7 | 0 / 7 |

* The generation is **deterministic and key-dependent**: the same key `(x0, y0)` reproduces the same S-box bit for bit;
  a 1e-14 change of the key gives a completely different S-box of the same quality (component NL 104, DU 8 for 22 of 23 keys).
* A closed-form Walsh-guided tabu search reaches component NL 104 in a few hundred swaps (seconds), where classical
  hill climbing / simulated annealing / GA plateau at 102 after 100 000 evaluations.
* Re-evaluation of ten published S-boxes with the same verified code shows that coordinate-average nonlinearity
  (106–111.5) hides component nonlinearities of 92–96 (LP 0.125–0.14); see `results/lit_reeval_results.txt`.

## The map (2D-LCChM)

```
x[n+1] = ( mu*x*(1-x) + beta*T_k(2y-1)     + lam*sin(pi*(x+y)     + theta_n) ) mod 1
y[n+1] = ( rho*y*(1-y^2) + beta*T_k(2x[n+1]-1) + lam*cos(pi*x[n+1]*y + theta_n) ) mod 1
T_k(u) = cos(k*arccos(u)),  theta_n = 2*pi*phi*n,  phi = (sqrt(5)-1)/2
```

Working regime: `mu=3.99, rho=2.59, k=5.3, lam=0.7, beta=1.0` (LE1 = 3.973, LE2 = -3.131; uniform ergodic orbit,
no periodic windows; NIST SP 800-22: all 15 tests). A hyperchaotic regime (LE2 > 0) exists for `lam < 0.5, beta < 0.13`
(see `figures/scan_hyper.png`).

## Repository layout

```
code/           all Python sources (run them from inside this folder)
sboxes/         the proposed S-box A and the second example B (txt, hex, csv, npy, json, C array)
results/        every result file quoted in the paper (plain text / json)
figures_paper/  the figures of the paper (Fig. 1–5, S1–S5; English labels, 300 dpi; produced by code/paper_figures.py)
figures/        working figures produced during the study (PNG)
```

Repository: `https://github.com/<yucelburhan>/lcchm-sbox`

### Key files in `code/`

| File | Purpose |
|---|---|
| `lccm_sbox.py` | map, Lyapunov exponents, raw S-box generation, all S-box metrics, AES reference |
| `lccm_guided.py` | Walsh-guided tabu search (closed-form swap scoring) → component NL 104 |
| `lccm_coord.py` | full design search (variant C1: NL 104, DU ≤ 8, basis-sum objective) |
| `lccm_dual.py` | output-basis selection (phase 3) and cost-function tuning experiments |
| `lccm_multikey.py` | multi-key statistics; produced S-box A |
| `lccm_keysens.py` | reproducibility and key-sensitivity experiment |
| `lccm_verify.py` | **independent brute-force verification of every metric** |
| `lccm_optimizers.py` | baseline HC / SA / tabu / GA comparison |
| `lccm_pipeline.py`, `lccm_du*.py` | earlier pipeline stages, control experiment (chaotic vs random start), DU-6 attempts |
| `lccm_figures.py`, `lccm_scan_*.py`, `lccm_regimes.py`, `lccm_chaos_metrics.py` | bifurcation, Lyapunov maps, hyperchaotic region, PE / SampEn / 0-1 test |
| `paper_figures.py` | regenerates the paper figures in `figures_paper/` (needs `scan_hyper.npz` from `lccm_scan_hyper.py` for Fig. 4) |
| `lccm_nist*.py` | NIST SP 800-22 (all 15 tests, self-validated on the NIST example vectors) |
| `lccm_image_enc*.py`, `lccm_cpa_demo.py`, `lccm_tag_sens.py` | image-encryption application, chosen-plaintext and tag-sensitivity demonstrations |
| `lit_extract_sboxes.py`, `lit_reeval.py` | extraction of published S-box tables from PDFs and their re-evaluation |

## Reproducing the main results

```bash
pip install -r requirements.txt
cd code

# 1. verify all metrics of S-box A (and AES) with the independent brute-force implementation (~1 min)
python lccm_verify.py lccm_multikey_best.npy

# 2. regenerate S-box A from its key (deterministic; ~10-15 min per key on 11 cores)
python lccm_multikey.py 22 2500        # 22 new keys + 6 previous; S-box A is the best of 28

# 3. reproducibility / key sensitivity (~15 min)
python lccm_keysens.py

# 4. map analysis figures, chaos metrics, NIST
python lccm_figures.py
python lccm_chaos_metrics.py
python lccm_nist.py && python lccm_nist_re.py && python lccm_nist_rest.py

# 5. literature re-evaluation (uses sboxes shipped in lit_sboxes_valid.json)
python lit_reeval.py
```

Key of S-box A (full double precision is required): `x0 = 0.0792210841394443`, `y0 = 0.09052887083835903`.

### Image-encryption experiments

The standard test images (Lena, Cameraman, Baboon, Peppers, House, Jetplane, Lake, Livingroom, Pirate,
Walkbridge, Woman) are **not redistributed** here for copyright reasons. Download them from
<https://www.imageprocessingplace.com/root_files_V3/image_databases.htm> ("standard test images", 512×512 grayscale TIFF),
place them in `code/`, then run `python lccm_image_enc_multi.py`. `lccm_image_enc.py` uses matplotlib's bundled
`grace_hopper.jpg` and needs nothing external.

### Published S-boxes used in the re-evaluation

`code/lit_sboxes_valid.json` contains the ten published 8-bit S-box tables re-evaluated in the paper
(Lambić 2014; Al Solami et al. 2018; Lu et al. 2019; Yang et al. 2021; Zheng & Bao 2022; Ali et al. 2022;
Mahboob et al. 2023; Abbasi et al. 2025; Banga et al. 2025; Razaq et al. 2026), transcribed from the
respective open-access papers (nine of them extracted automatically from the PDFs and verified cell by cell).
The papers themselves are not included; see the reference list of the paper.

## Notes on metric definitions

* **Component NL**: minimum nonlinearity over all 255 non-zero linear combinations of the output bits
  (the standard S-box nonlinearity; equals 128 − max|W|/2). LP = (128 − component NL)/256.
* **Coordinate NL**: nonlinearity of each of the 8 output-bit Boolean functions (min / avg / max);
  this is what most chaos-based S-box papers report as "NL".
* NPCR / UACI critical values are image-size dependent (Wu, Noonan & Agaian 2011); 512×512: NPCR > 99.5893 %,
  UACI 33.3730–33.5541 %; 256×256: NPCR > 99.5693 %, UACI 33.2824–33.6447 %.

## Citation

```
[to be added after acceptance]
```

## License

MIT (see `LICENSE`). Environment used for the reported results: Python 3.14, NumPy 2.5, 12-core Windows 11.
