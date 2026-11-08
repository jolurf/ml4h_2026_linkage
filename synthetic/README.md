# Synthetic reproduction — CNS validation, cadastral entropy, and Fellegi-Sunter

Accompanies the ML4H 2026 paper *Auditing Identity Noise and Linkage
Degradation at National Scale*. Two scripts, each covering a formula from the
paper's appendices, both on **synthetic or already-published** values — no
real patient data.

**Contains no real patient data, no real CNS, and no dependency on
DATASUS production data.** `validate_synthetic_cns.py` generates
identifiers with a pseudo-random generator under a fixed seed
(`SEED = 20260907`); `entropy_and_fs_example.py` uses fictitious
identifiers (part 1) and published m/u values (part 2) — neither script
reads any file.

## Running

From the repository root:

```bash
pip install validate-docbr
python synthetic/validate_synthetic_cns.py
python synthetic/entropy_and_fs_example.py   # does not need validate-docbr
```

No GPU, no network beyond installing the package above, a few seconds in
total.

## `validate_synthetic_cns.py` — check digit (Appendix B.1)

1. Generates 300,000 synthetic CNS values, valid by construction (two
   families: CPF/PIS-derived and provisional — see the algorithm below),
   including a subset specifically constructed to exercise the check
   digit's `dv==10` special case (50,053 values fall in that branch).
2. Validates each one with two independent implementations:
   - `check_digit_valid`/`_complete_pis`, ported line-for-line from the
     function the authors' pipeline uses at national scale (identifiers
     translated to English; logic unchanged);
   - [`validate-docbr`](https://github.com/alvarofpp/validate-docbr), an
     actively maintained third-party Python library.
3. Runs two negative controls: the 10 single-repeated-digit placeholders
   (must be rejected by both implementations) and 1,000 valid CNS each
   corrupted in one random digit (both implementations must agree on each).
4. Prints the agreement between the two implementations over the 301,010
   checks: expected 300,000/300,000 on the positive corpus and zero
   disagreements overall.

## `entropy_and_fs_example.py` — cadastral entropy (Appendix B.2) and weight of evidence (Appendix C)

1. **Cadastral entropy**: the same Shannon-entropy formula used in the
   pipeline's cadastral-entropy module, applied to three fictitious
   identifiers illustrating the regimes behind Appendix B.2 — a stable
   identity, occasional drift against a dominant identity (low entropy
   despite 2 identities), and an even 50/50 alternation (maximum entropy).
2. **Fellegi-Sunter weight**: `weight_bits()`, ported from the pipeline's
   weight-of-evidence function, run over the national single-field m and u
   of the Fellegi–Sunter table (Appendix C, Table 11; $u$ excludes
   self-pairs): birth date 0.973 / 3.29e-5, sex 0.989 / 0.515, municipality
   0.924 / 7.65e-3. It reproduces the per-field weights (14.85, 0.94 and
   6.92 bits), then compares, for birth date + sex + municipality, the sum
   of the per-field weights (22.71 bits) with the weight from the key
   measured jointly (m 0.895, u 1.32e-7: 22.69 bits). Nationally the joint
   $u$ is 1.02× the product of the single-field $u$ (derived here and
   checked by an `assert`), so the two weights agree. In the nephroblastoma
   pool the published ratio is 30.4× without self-pairs (7.7×–30.4×
   depending on how likely duplicate children are excluded), so summing the
   per-field weights overstates the evidence of a cohort match by about
   log2 of that ratio, 4.9 bits (2.9–4.9 bits over the range). The cohort
   ratios themselves are computed from the cohort by
   `analysis/fs_diagnostico_u_m.py` (`D-RATIO-NOSELF` and
   `D-RATIO-ALLCNES` in `numbers/provenance.md`).

## Why this exists

The paper's appendices describe validations and formulas that are otherwise
only demonstrated inside the authors' pipeline, which is not released (see the
paper's "Data and Code Availability" section). This folder provides an
independent, executable artifact for three of them: check digit, cadastral
entropy, and Fellegi-Sunter weight of evidence.
