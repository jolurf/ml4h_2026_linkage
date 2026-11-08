#!/usr/bin/env python3
"""Supplementary material (tier 2): demonstrates, on toy or already-published
data, the other two formulas from the paper's appendices cited in the "Data and
Code Availability" section -- cadastral entropy and Fellegi-Sunter weight of
evidence. Depends on neither `dados/` nor `paths.py`.

Two parts, each explicit about what is REAL (published in the paper) and
what is ILLUSTRATIVE (toy numbers, only to show the mechanism):

  1. Cadastral entropy -- formula identical to the production pipeline's
     cadastral-entropy module (Shannon entropy, in bits), applied to a
     handful of FICTITIOUS identifiers constructed to exercise the three
     qualitative regimes behind Appendix B.2 of the paper: a stable identity
     (entropy 0), occasional drift against a dominant identity (low entropy
     despite >1 identity), and an even split (high entropy). None of the
     numbers here are measured -- everything is constructed to illustrate
     the formula.

  2. Fellegi-Sunter weight -- `weight_bits()` ported line-for-line from
     the production pipeline's weight-of-evidence function, run over
     `PUBLISHED_MU_FIELD`, the national single-field m and u of the paper's
     Fellegi-Sunter table (Appendix C, Table 11; u excludes self-pairs).
     Reproduces the per-field weights; then, for the birth date + sex +
     municipality key, compares the sum of the per-field weights (which
     assumes independent fields) with the weight from the jointly measured
     m and u. Nationally the joint u is 1.02x the product of the single-field
     u, so the two weights agree; in the nephroblastoma pool the published
     ratio is 30.4x (7.7x-30.4x depending on how likely duplicate children
     are excluded), and the sum overstates the evidence of a cohort match by
     about log2 of that ratio.

Usage:
    python synthetic/entropy_and_fs_example.py
"""
from __future__ import annotations

import math
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Cadastral entropy -- formula identical to the production pipeline's
#    per-partition cadastral-entropy aggregation
#    (-SUM((n_i/n_total) * log2(n_i/n_total)))
# ---------------------------------------------------------------------------

def entropy_bits(identities: list[tuple]) -> float:
    """`identities`: one entry per record of the CNS (date of birth, sex,
    municipality of residence) -- may repeat. Identical to the real
    aggregation, just done in memory instead of SQL."""
    counts = Counter(identities)
    n_total = sum(counts.values())
    return -sum((n / n_total) * math.log2(n / n_total) for n in counts.values())


FICTITIOUS_CNS = {
    # (dob, sex, municipality) repeated = same identity across all records
    "STABLE_CNS (10 records, 1 identity)":
        [("1985-03-12", "F", "355030")] * 10,

    # 1 record with a typo among 9 correct ones -- low entropy despite 2
    # identities, because one dominates (Appendix B.2: drift against a
    # dominant identity)
    "OCCASIONAL_TYPO_CNS (10 records, 1 typo)":
        [("1985-03-12", "F", "355030")] * 9 + [("1985-03-21", "F", "355030")] * 1,

    # 50/50 alternation between two identities -- maximum entropy (1 bit)
    "50_50_ALTERNATION_CNS (10 records, even split)":
        [("1985-03-12", "F", "355030")] * 5 + [("1990-07-04", "F", "310620")] * 5,
}


def demo_entropy() -> None:
    print("=" * 72)
    print("1. CADASTRAL ENTROPY (fictitious data, only to illustrate the formula)")
    print("=" * 72)
    for name, records in FICTITIOUS_CNS.items():
        n_identities = len(set(records))
        e = entropy_bits(records)
        print(f"  {name}")
        print(f"      {n_identities} distinct identity(ies) -> {e:.3f} bits")
    print()
    print("  Note: the CNS with 1 typo has >1 identity but low entropy (the "
          "dominant identity pulls the distribution down); the 50/50 "
          "alternation one has the SAME identity count but maximum entropy "
          "-- which is why Appendix B.2 of the paper reports both the mean "
          "entropy and the share of CNS with more than one identity.")


# ---------------------------------------------------------------------------
# 2. Fellegi-Sunter weight -- weight_bits() ported from the production
#    pipeline's weight-of-evidence function, run over the national m/u of
#    the paper's Fellegi-Sunter table (Appendix C, Table 11).
# ---------------------------------------------------------------------------

# National BI, single fields: (m, u). u = sum n_i(n_i-1) / N(N-1), self-pairs
# excluded; m = agreement among records of the same CNS, weighted by patient.
# Published values, not invented here.
PUBLISHED_MU_FIELD: dict[str, tuple[float, float]] = {
    "date_of_birth": (0.973, 3.29e-5),
    "sex": (0.989, 0.515),
    "municipality_of_residence": (0.924, 7.65e-3),
}

# National BI, birth date + sex + municipality measured as one key.
PUBLISHED_M_JOINT = 0.895            # measured jointly (not the product of the m above)
PUBLISHED_U_JOINT = 1.32e-7          # self-pairs excluded
PUBLISHED_RATIO_NATIONAL = 1.02      # joint u / product of the single-field u

# Nephroblastoma pool, same key: joint u / independent product, self-pairs
# excluded. The range depends on which pairs of distinct CNS are excluded as
# likely the same child recorded twice: none (30.4x) up to every pair ever
# seen at the same facility (7.7x).
PUBLISHED_RATIO_COHORT = 30.4
PUBLISHED_RATIO_COHORT_RANGE = (7.7, 30.4)


def weight_bits(fields: tuple[str, ...], mu: dict) -> float | None:
    """Ported line-for-line from the production pipeline's weight-of-
    evidence function. Sum of log2(m/u) over the fields of the key --
    naive sum, assumes field independence."""
    total = 0.0
    for field in fields:
        m, u = mu.get(field, (None, None))
        if not m or not u:
            return None
        total += math.log2(m / u)
    return round(total, 2)


def demo_fellegi_sunter() -> None:
    print()
    print("=" * 72)
    print("2a. WEIGHT PER ISOLATED FIELD (national m/u, Appendix C, Table 11)")
    print("=" * 72)
    for field in PUBLISHED_MU_FIELD:
        w = weight_bits((field,), PUBLISHED_MU_FIELD)
        print(f"  {field:<28} weight = {w} bits")

    print()
    print("=" * 72)
    print("2b. SUM OF FIELDS vs. JOINT KEY, NATIONAL (Appendix C)")
    print("=" * 72)
    w_sum = weight_bits(tuple(PUBLISHED_MU_FIELD), PUBLISHED_MU_FIELD)
    m_product = math.prod(m for m, _ in PUBLISHED_MU_FIELD.values())
    u_product = math.prod(u for _, u in PUBLISHED_MU_FIELD.values())
    w_joint = math.log2(PUBLISHED_M_JOINT / PUBLISHED_U_JOINT)
    ratio = PUBLISHED_U_JOINT / u_product
    print(f"  sum of the 3 per-field weights: {w_sum:.2f} bits "
          f"(product of m {m_product:.3f}, product of u {u_product:.3e})")
    print(f"  joint key (m {PUBLISHED_M_JOINT}, u {PUBLISHED_U_JOINT:.2e}): {w_joint:.2f} bits")
    print(f"  joint u / product of u = {ratio:.2f}x (paper: {PUBLISHED_RATIO_NATIONAL}x): "
          f"nationally the three fields are close to independent, and the two "
          f"weights differ by {abs(w_sum - w_joint):.2f} bits")
    assert abs(ratio - PUBLISHED_RATIO_NATIONAL) < 0.01, ratio

    print()
    print("=" * 72)
    print("2c. THE SAME KEY IN THE NEPHROBLASTOMA POOL (Appendix C)")
    print("=" * 72)
    lo, hi = PUBLISHED_RATIO_COHORT_RANGE
    print(f"  joint u / independent product = {PUBLISHED_RATIO_COHORT}x, self-pairs "
          f"excluded ({lo}x-{hi}x depending on how likely duplicate children are excluded)")
    print(f"  summing per-field weights uses the product of u, so it overstates the "
          f"evidence of a cohort match by about log2({PUBLISHED_RATIO_COHORT}) = "
          f"{math.log2(PUBLISHED_RATIO_COHORT):.1f} bits "
          f"({math.log2(lo):.1f}-{math.log2(hi):.1f} bits over the range)")


if __name__ == "__main__":
    demo_entropy()
    demo_fellegi_sunter()
