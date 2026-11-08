#!/usr/bin/env python3
"""Sensibilidade da sobrevida à qualidade do match RHC↔SUS.

O join que ancora a coorte não tem CNS. Cada paciente carrega a regra que o
ligou (`TIPO_MATCH`: CNES do hospital + sexo, ou município + sexo), a diferença
entre as datas de diagnóstico (`diff_dias`, até 75) e entre os anos de nascimento
(`diff_anos`, 0 ou 1). Quanto mais estrito o match, menor a chance de ser outra
criança. Aqui a OS em 5 anos e o HR do estádio IV são recalculados restringindo
a coorte a matches cada vez mais estritos — e ao complemento de cada estrato.

`concordancia_status` NÃO entra: "Erro (Discordante)" é divergência de status
vital entre RHC e SUS, o que condiciona no desfecho.

Saídas em `experimento/` (+ outputs/):
    tabela_sensibilidade_match_rhc.csv
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1] / "src"))
import gerar_relatorio_wilms as G  # noqa: E402
from cenarios_degradacao_linkage import _hr_estadio_iv  # noqa: E402

EXP = AQUI / "experimento"
AGREGADOS = AQUI.parents[1] / "outputs"


def medir(d: pd.DataFrame, rotulo: str) -> dict:
    s60, lo, hi = G.os_com_ic(d["T_meses"], d["E_final"], G.TAU)
    hr, hlo, hhi, n_cox = _hr_estadio_iv(d, "T_meses", "E_final")
    if not np.isfinite(hhi) or hhi / max(hlo, 1e-12) > 1e3:  # estrato pequeno: Cox não identificado
        hr = hlo = hhi = np.nan
    return {"estrato": rotulo, "n": len(d), "obitos": int(d["E_final"].sum()),
            "os_5a_pct": s60, "os_ic_lo": lo, "os_ic_hi": hi,
            "hr_estadio_iv": hr, "hr_ic_lo": hlo, "hr_ic_hi": hhi, "n_cox": n_cox}


def main() -> None:
    d, _ = G.carregar_coorte(colapsar=True)
    estratos = {
        "match pelo CNES do hospital (não por município)": d["TIPO_MATCH"] == "CNES_HOSPITAL",
        "diferença de data de diagnóstico ≤ 7 dias": d["diff_dias"] <= 7,
        "diferença de data de diagnóstico = 0": d["diff_dias"] == 0,
        "mesmo ano de nascimento (diff_anos = 0)": d["diff_anos"] == 0,
        "CNES ∧ ≤ 7 dias ∧ mesmo ano de nascimento": (d["TIPO_MATCH"] == "CNES_HOSPITAL") & (d["diff_dias"] <= 7) & (d["diff_anos"] == 0),
    }
    linhas = [medir(d, "todos (referência)")]
    for rot, m in estratos.items():
        linhas.append(medir(d[m], rot))
        linhas.append(medir(d[~m], "complemento: " + rot))
    t = pd.DataFrame(linhas)
    ref = t.iloc[0]
    assert (ref.n, ref.obitos) == (2684, 474) and abs(ref.os_5a_pct - 80.49) < 0.05 and abs(ref.hr_estadio_iv - 2.61) < 0.01, ref
    print(t.round(2).to_string(index=False))
    EXP.mkdir(exist_ok=True)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    t.to_csv(EXP / "tabela_sensibilidade_match_rhc.csv", index=False)
    shutil.copy2(EXP / "tabela_sensibilidade_match_rhc.csv", AGREGADOS / "tabela_sensibilidade_match_rhc.csv")
    print(f"✅ {EXP / 'tabela_sensibilidade_match_rhc.csv'}")


if __name__ == "__main__":
    main()
