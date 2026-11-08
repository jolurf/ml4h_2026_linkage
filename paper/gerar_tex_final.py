"""Gera paper/ml4h_final.tex a partir de paper/ml4h_final.tex.in.

Cada {{TAG}} do modelo é trocado pelo valor em numbers/numbers.json (o mapa
tag → valor → script → CSV, ver numbers/provenance.md), escapado para LaTeX.
Falha se sobrar tag sem número.

    python paper/gerar_tex_final.py
    cd paper && pdflatex ml4h_final && bibtex ml4h_final && pdflatex ml4h_final && pdflatex ml4h_final

(compilar exige a classe do template ML4H 2026, jmlr.cls, que não vem neste repositório)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
TAG = re.compile(r"\{\{([A-Za-z0-9\-]+)\}\}")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def latex(v: str) -> str:
    v = re.sub(r"×10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)", lambda m: r"$\times10^{" + m.group(1).translate(SUP) + "}$", v)
    v = v.replace("%", r"\%").replace("≤", r"$\leq$").replace("≥", r"$\geq$").replace("×", r"$\times$")
    return v


def main() -> int:
    num = json.loads((AQUI.parent / "numbers" / "numbers.json").read_text())
    src = (AQUI / "ml4h_final.tex.in").read_text()
    faltam = sorted({t for t in TAG.findall(src) if t not in num})
    if faltam:
        print("❌ tags sem número:", ", ".join(faltam))
        return 1
    out = TAG.sub(lambda m: latex(num[m.group(1)]["valor"]), src)
    (AQUI / "ml4h_final.tex").write_text(out)
    print(f"✅ ml4h_final.tex ({len(set(TAG.findall(src)))} tags)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
