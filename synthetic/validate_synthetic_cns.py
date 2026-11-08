#!/usr/bin/env python3
"""Independent reproduction of the CNS check-digit validation reported in
Appendix B.1 of the paper.

Runs entirely on SYNTHETIC identifiers -- no real DATASUS data enters this
script, no dependency on the production pipeline's data directories or
path-resolution module. This is the anonymized supplementary material
referenced in the paper's "Data and Code Availability" section.

Two independent implementations are cross-checked, CNS by CNS:

  (1) `check_digit_valid` / `_complete_pis`, ported line-for-line from the
      pure (no I/O, no cache decryption) part of the production pipeline's
      own check-digit function -- the function actually used, at national
      scale, to validate 3.42 billion well-formed CNS values. Names are
      translated to English here for readability; the algorithm and logic
      are unchanged.
  (2) `validate_docbr.CNS().validate()`, an actively maintained third-party
      Python library (https://github.com/alvarofpp/validate-docbr), whose
      algorithm (`_generate_first_case`) was read and confirmed to be the
      SAME algorithm (weights 15..5, the dv==10 branch adding 2 to the
      weighted sum with suffix '001', not '000') before this script was
      written.

The algorithm itself (modulus 11, two families by prefix) had already been
manually cross-checked against the e-SUS APS Java reference documentation
(integracao.esusaps.bridge.ufsc.tech/v211/docs/algoritmo_CNS.html) on
2026-08-24 -- see the docstring of the production pipeline's check-digit
module. This script reproduces that same algorithm at scale (300,000
synthetic CNS), automated and re-executable, which the earlier manual
check was not.

Synthetic corpus composition (300,000 CNS, valid by construction, fixed
seed -- the numbers below are exactly reproduced on every run):
  - 250,000 CPF/PIS family (prefix 1 or 2), uniformly random 11-digit
    prefix, check digit completed by the algorithm. By construction,
    ~1/11 incidentally fall into the dv==10 branch.
  - 27,255 of the SAME family, specifically constructed by rejection
    sampling to land in the special dv==10 branch (weighted_sum % 11 == 1)
    -- the branch a naive implementation historically got wrong (see the
    `_complete_pis` docstring).
  - 22,745 provisional family (prefix 7, 8, or 9), 14 random digits + the
    15th solved to close weighted_sum % 11 == 0 -- never hits the dv==10
    branch (exclusive to the CPF/PIS family).
  Total: 300,000. Adding the 27,255 deliberately constructed cases to the
  ones that incidentally land in that branch from the general 250,000
  pool, the real total of CNS in the corpus that exercise the dv==10
  branch is reported by the script's own output ("of which dv==10") -- it
  is not a number fixed in advance, it is counted after generation, with
  the same seed always reproducing the same value.

One more negative control, outside the count of 300,000 (these are not
"valid synthetic CNS", they are the opposite sanity check): the 10
single-repeated-digit placeholders (`000000000000000` .. `999999999999999`),
which must be rejected by both implementations -- this is the
placeholder sanity check of the paper's CNS validation (0% may pass),
reproduced here on a synthetic universe instead of the national BI.

Usage:
    pip install validate-docbr
    python synthetic/validate_synthetic_cns.py
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

try:
    from validate_docbr import CNS as _DocBrCNS
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script depends on the external reference implementation:\n"
        "    pip install validate-docbr"
    ) from exc

CNS_WIDTH = 15
SEED = 20260907  # fixed, so the report is byte-for-byte re-executable

N_PIS_GENERAL = 250_000
N_PIS_DV10 = 27_255
N_PROVISIONAL = 22_745
N_TOTAL = N_PIS_GENERAL + N_PIS_DV10 + N_PROVISIONAL  # 300_000


# ---------------------------------------------------------------------------
# Implementation (1) -- ported line-for-line from the production
# pipeline's check-digit module (identifiers translated to English; the
# decryption step, which is specific to the BI cache format and not part
# of the CNS algorithm itself, is omitted).
# ---------------------------------------------------------------------------

def check_digit_valid(d: str) -> bool:
    """Official CNS algorithm (modulus 11). `d` is already plaintext (15
    decimal digits) -- unlike the production version, which received the
    encrypted CNS from the BI cache, there is no encryption here at all,
    everything is synthetic."""
    if len(d) != CNS_WIDTH or not d.isdigit():
        return False
    if d[0] in "12":
        pis = d[:11]
        return _complete_pis(pis) == d
    if d[0] in "789":
        total = sum(int(c) * weight for c, weight in zip(d, range(15, 0, -1)))
        return total % 11 == 0
    return False


def _complete_pis(pis: str) -> str:
    """CNS derived from CPF/PIS (prefix 1 or 2): weighted sum (weights
    15..5) of the 11-digit PIS, check digit by remainder mod 11, suffix
    '000' in the normal case. Special case dv==10: add 2 to the WEIGHTED
    SUM (not to the PIS value itself) and recompute; suffix becomes '001'."""
    def weighted_sum(p: str) -> int:
        return sum(int(c) * weight for c, weight in zip(p, range(15, 4, -1)))

    total = weighted_sum(pis)
    remainder = total % 11
    dv = 11 - remainder
    if dv == 11:
        dv = 0
    if dv == 10:
        total += 2
        remainder = total % 11
        dv = 11 - remainder
        return f"{pis}001{dv}"
    return f"{pis}000{dv}"


# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------

def _random_pis(rng: random.Random) -> str:
    first = rng.choice("12")
    rest = "".join(rng.choice("0123456789") for _ in range(10))
    return first + rest


def generate_pis_general(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        pis = _random_pis(rng)
        out.append(_complete_pis(pis))
    return out


def generate_pis_dv10(rng: random.Random, n: int) -> list[str]:
    """Rejection sampling: draws PIS values until one lands in the
    total%11==1 branch (which forces dv==10 inside `_complete_pis`),
    repeats until `n` are collected."""
    out = []
    while len(out) < n:
        pis = _random_pis(rng)
        total = sum(int(c) * weight for c, weight in zip(pis, range(15, 4, -1)))
        if total % 11 == 1:
            out.append(_complete_pis(pis))
    return out


def generate_provisional(rng: random.Random, n: int) -> list[str]:
    out = []
    while len(out) < n:
        first = rng.choice("789")
        middle = [rng.choice("0123456789") for _ in range(13)]
        partial = [first] + middle
        partial_sum = sum(int(c) * weight for c, weight in zip(partial, range(15, 1, -1)))
        for last in "0123456789":
            if (partial_sum + int(last)) % 11 == 0:
                out.append("".join(partial) + last)
                break
        # if no final digit closes the sum (can happen, 1 in 11
        # combinations has no solution in 0..9), discard and retry
    return out


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

@dataclass
class Result:
    total: int = 0
    agree: int = 0
    both_valid: int = 0
    disagree: list[str] = field(default_factory=list)

    @property
    def both_invalid(self) -> int:
        return self.agree - self.both_valid


def validate_batch(cns_list: list[str], ref: _DocBrCNS) -> Result:
    r = Result(total=len(cns_list))
    for cns in cns_list:
        a = check_digit_valid(cns)
        b = ref.validate(cns)
        if a == b:
            r.agree += 1
            if a:
                r.both_valid += 1
        else:
            r.disagree.append(cns)
    return r


def main() -> None:
    rng = random.Random(SEED)
    ref = _DocBrCNS()  # single instance, reused across all checks

    print(f"Generating {N_TOTAL:,} synthetic CNS (seed={SEED})...")
    corpus = (
        generate_pis_general(rng, N_PIS_GENERAL)
        + generate_pis_dv10(rng, N_PIS_DV10)
        + generate_provisional(rng, N_PROVISIONAL)
    )
    assert len(corpus) == N_TOTAL, f"expected {N_TOTAL}, generated {len(corpus)}"

    n_dv10_actual = sum(1 for c in corpus if c[0] in "12" and c[11:14] == "001")
    print(f"  {len(corpus):,} generated; {n_dv10_actual:,} actually fall into the "
          f"dv==10 branch (suffix '001')")

    print("Validating the positive corpus (300,000 CNS valid by construction) "
          "against validate-docbr...")
    r_pos = validate_batch(corpus, ref)

    print("Validating negative control (10 repeated-digit placeholders)...")
    placeholders = [str(d) * 15 for d in range(10)]
    n_placeholders_pass = sum(
        1 for c in placeholders if check_digit_valid(c) or ref.validate(c)
    )

    print("Validating extra negative control (1,000 valid CNS corrupted "
          "in 1 random digit)...")
    corrupted = []
    for cns in rng.sample(corpus, 1000):
        pos = rng.randrange(CNS_WIDTH)
        new_digit = rng.choice([d for d in "0123456789" if d != cns[pos]])
        corrupted.append(cns[:pos] + new_digit + cns[pos + 1:])
    r_corrupted = validate_batch(corrupted, ref)

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Positive corpus:       {r_pos.total:,} synthetic CNS valid by construction")
    print(f"  agreement:           {r_pos.agree:,}/{r_pos.total:,} "
          f"({100 * r_pos.agree / r_pos.total:.4f}%)")
    print(f"  both say valid:      {r_pos.both_valid:,}/{r_pos.total:,} "
          f"({100 * r_pos.both_valid / r_pos.total:.4f}%)")
    print(f"  disagreements:       {len(r_pos.disagree)}")
    print(f"  of which dv==10:     {n_dv10_actual:,}")
    print()
    print(f"Placeholders (10, single repeated digit):")
    print(f"  pass rate:           {n_placeholders_pass}/10 "
          f"({100 * n_placeholders_pass / 10:.1f}%) -- expected 0%")
    print()
    print(f"Corrupted CNS (1,000, 1 digit changed):")
    print(f"  both reject:         {r_corrupted.both_invalid:,}/1000")
    print(f"  disagreements:       {len(r_corrupted.disagree)}")

    if r_pos.disagree or n_placeholders_pass or r_corrupted.disagree:
        print()
        print("WARNING: the two implementations disagreed somewhere -- this "
              "CHANGES the text of Appendix B.1, which can no longer say "
              "'zero disagreement'. Investigate the cases in `disagree` "
              "before updating the paper.")
    else:
        print()
        print("Zero disagreements across all three checks -- confirms the "
              "number to use in Appendix B.1.")


if __name__ == "__main__":
    main()
