#!/usr/bin/env python3
"""Validação externa, pelo RHC, das duas arestas de óbito que entram na coorte.

A coorte é ancorada no RHC. O RHC liga ao SIA (identidade com CNS); o SIH só
entra porque a cascata R1–R4 ligou a AIH a uma identidade do SIA. Um erro nessa
cascata tem dois destinos: a AIH não vinculada fica fora (3 386 de 22 809 no
projeto, `sem_identidade`) e nunca entra na análise; a AIH vinculada ao paciente
errado ENTRA, e o que ela pode mudar no desfecho é a marca de óbito (`MORTE`)
que a Camada 1 lê. O `DATAOBITO` do RHC é registrado pelo registro hospitalar,
fora do SUS, e serve de conferência independente das duas arestas:

1. **SIA/SIH → SIM** (Camada 2): quantos óbitos do RHC o linkage ao SIM achou.
2. **SIH → paciente** (Camada 1 vinda de AIH): quantos óbitos que dependem só de
   uma AIH vinculada pela cascata o RHC confirma.

E duas sensibilidades que censuram o que ninguém confirma.

Saídas em `experimento/` (copiadas para outputs/):
    tabela_rhc_x_sus.csv            cruzamento RHC × flag do pipeline
    tabela_obitos_c1_por_fonte.csv  óbitos C1 por origem (SIA, SIH, ambos) e confirmação
    tabela_sensibilidade_sih.csv    OS/HR censurando óbitos só-SIH não confirmados
"""
from __future__ import annotations

import shutil
from pathlib import Path

import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import gerar_relatorio_wilms as G  # noqa: E402
from aux_recuperacao import ic_wilson  # noqa: E402
from cenarios_degradacao_linkage import _hr_estadio_iv

AQUI = Path(__file__).resolve().parent
EXP = AQUI / "experimento"
AGREGADOS = AQUI.parents[1] / "outputs"


def _wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = ic_wilson(np.array([float(k)]), np.array([float(n)]))
    return 100 * float(np.ravel(lo)[0]), 100 * float(np.ravel(hi)[0])


def _gravar(df: pd.DataFrame, nome: str) -> None:
    df.to_csv(EXP / nome, index=False)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXP / nome, AGREGADOS / nome)
    print(f"✅ {EXP / nome}")


def main() -> None:
    d, _ = G.carregar_coorte(colapsar=True)
    con = duckdb.connect()
    con.register("d", d[["id_paciente", "cat_estadiamento", "T_meses", "E_final", "E_full",
                         "evento_obito_sus", "evento_obito_inca"]])
    con.execute(f"""CREATE TABLE fonte AS SELECT id_unico,
        SUM((subbase = 'SIH' AND obito = '1')::INT)                          AS aih_morte,
        SUM((subbase = 'SIH' AND obito = '1' AND NOT link_chave_ambigua)::INT) AS aih_morte_nao_amb,
        SUM((subbase <> 'SIH' AND obito = '1')::INT)                         AS sia_obito
        FROM '{AQUI}/df_final_scored.parquet' GROUP BY id_unico""")
    j = con.execute(f"""SELECT d.*, s.link_obito_status,
            COALESCE(f.aih_morte, 0) aih_morte, COALESCE(f.aih_morte_nao_amb, 0) aih_morte_nao_amb,
            COALESCE(f.sia_obito, 0) sia_obito
        FROM d LEFT JOIN '{AQUI}/obitos_scored.parquet' s ON s.id_unico = d.id_paciente
               LEFT JOIN fonte f ON f.id_unico = d.id_paciente""").df()
    assert len(j) == len(d) == 2684

    # 1. RHC × pipeline (seguimento completo e horizonte de 5 anos)
    linhas = []
    for horizonte, E_sus, E_rhc in [
        ("completo", j["evento_obito_sus"] == 1, j["evento_obito_inca"] == 1),
        ("5 anos", (j["evento_obito_sus"] == 1) & (j["E_final"] == 1),
         (j["evento_obito_inca"] == 1) & (j["E_final"] == 1)),
    ]:
        n_rhc, k = int(E_rhc.sum()), int((E_rhc & E_sus).sum())
        lo, hi = _wilson(k, n_rhc)
        linhas.append({"horizonte": horizonte,
                       "nenhum": int((~E_rhc & ~E_sus).sum()), "so_rhc": int((E_rhc & ~E_sus).sum()),
                       "so_pipeline": int((~E_rhc & E_sus).sum()), "ambos": k,
                       "obitos_rhc": n_rhc, "pct_rhc_achados_pelo_pipeline": 100 * k / n_rhc,
                       "ic_lo": lo, "ic_hi": hi})
    rhc = pd.DataFrame(linhas)
    c = rhc.iloc[0]
    assert (c.nenhum, c.so_rhc, c.so_pipeline, c.ambos) == (2195, 113, 146, 230), "cruzamento mudou"
    print(rhc.to_string(index=False))
    _gravar(rhc, "tabela_rhc_x_sus.csv")

    # 2. Óbitos C1 por origem da marca de óbito
    c1 = j[j["link_obito_status"] == "obito_interno_C1"].copy()
    c1["origem"] = np.select(
        [(c1.aih_morte > 0) & (c1.sia_obito == 0), (c1.aih_morte == 0) & (c1.sia_obito > 0),
         (c1.aih_morte > 0) & (c1.sia_obito > 0)], ["so_SIH", "so_SIA", "ambos"], "nenhum")
    linhas = []
    for origem, g in c1.groupby("origem"):
        for horizonte, gg in [("completo", g), ("5 anos", g[g["E_final"] == 1])]:
            k = int((gg["evento_obito_inca"] == 1).sum())
            lo, hi = _wilson(k, len(gg))
            linhas.append({"origem": origem, "horizonte": horizonte, "n": len(gg),
                           "confirmados_rhc": k, "pct_confirmados": 100 * k / len(gg) if len(gg) else np.nan,
                           "ic_lo": lo, "ic_hi": hi,
                           "so_chave_ambigua": int(((gg.aih_morte > 0) & (gg.aih_morte_nao_amb == 0)
                                                    & (gg.sia_obito == 0)).sum())})
    fontes = pd.DataFrame(linhas)
    print(fontes.to_string(index=False))
    _gravar(fontes, "tabela_obitos_c1_por_fonte.csv")

    # 3. Sensibilidade: censurar óbitos que dependem só de AIH e que o RHC não confirma
    so_sih = (j["link_obito_status"] == "obito_interno_C1") & (j.aih_morte > 0) & (j.sia_obito == 0)
    nao_conf = so_sih & (j["evento_obito_inca"] != 1)
    so_amb = so_sih & (j.aih_morte_nao_amb == 0) & (j["evento_obito_inca"] != 1)
    linhas = []
    for nome, censurar in [("0_referencia", pd.Series(False, index=j.index)),
                           ("S1_censura_so_SIH_ambiguo_nao_confirmado", so_amb),
                           ("S2_censura_so_SIH_nao_confirmado", nao_conf)]:
        E = np.where(censurar, 0, j["E_final"])
        s60, lo, hi = G.os_com_ic(j["T_meses"], pd.Series(E), G.TAU)
        dd = j[["cat_estadiamento"]].copy()
        dd["T"], dd["E"] = j["T_meses"].values, E
        hr, hlo, hhi, _ = _hr_estadio_iv(dd, "T", "E")
        linhas.append({"cenario": nome, "obitos_censurados": int((censurar & (j["E_final"] == 1)).sum()),
                       "obitos": int(E.sum()), "os_5a_pct": s60, "os_ic_lo": lo, "os_ic_hi": hi,
                       "hr_estadio_iv": hr, "hr_ic_lo": hlo, "hr_ic_hi": hhi})
    sens = pd.DataFrame(linhas)
    assert abs(sens.loc[0, "hr_estadio_iv"] - 2.61) < 0.01, "referência não reproduz a Tabela 1"
    print(sens.round(2).to_string(index=False))
    _gravar(sens, "tabela_sensibilidade_sih.csv")


if __name__ == "__main__":
    main()
