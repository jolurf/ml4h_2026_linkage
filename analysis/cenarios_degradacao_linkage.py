#!/usr/bin/env python3
"""Tabela dos 4 Cenários de degradação de linkage — reaproveita a coorte oficial
(`gerar_relatorio_wilms.carregar_coorte`) e os artefatos de linkage já
pontuados (`df_final_scored.parquet`, `obitos_scored.parquet`), sem
recomputar nenhum linkage do zero. Só reclassifica desfechos.

**Desenho: tempo de seguimento fixo, só o desfecho muda.** Em todos os
cenários (exceto o 3, que muda o grão), `T_meses` vem inalterado da coorte
oficial — a data de última atividade não é re-derivada por cenário. Isso
isola o efeito de CADA degradação isoladamente (ausência de linkage ao SIM,
contaminação por chave ambígua) do efeito de também perder o rastro
temporal, que exigiria reconstruir a coorte de outra fonte-base — fora do
escopo desta tabela. Está documentado aqui, não escondido.

**Cenário 0 (referência)**: a própria coorte oficial —
`d["T_meses"]`/`d["E_final"]` de `carregar_coorte()`, sem modificação.
2 684 pacientes, 489 óbitos, OS 5 anos 80,5% (relatório da coorte).

**Cenário 1 (base única / claims ingênuo)**: óbito só quando a Camada 1 do
funil de mortalidade capturou (`obitos_scored.parquet`,
`link_obito_status == 'obito_interno_C1'` — "o próprio registro SIA/SIH
marca óbito", sem qualquer linkage ao SIM) **e** ignorando também o
`DATAOBITO` do RHC (que é um registro clínico externo, não uma claim de
internação — um cenário "só SIH" não o teria). Mede quanto óbito
desaparece — vira censura à direita — sem NENHUMA fonte externa de
mortalidade linkada.

**Cenário 2 (contaminação por chave ambígua, medida por remoção)** — nome
técnico `2_ambiguidade_removida`, de propósito não "ativa": o Cenário 0
JÁ inclui os óbitos de chave ambígua (a convenção do pipeline é manter
vínculos ambíguos com desempate determinístico, não descartá-los — Seção 6
do documento principal, "Ambiguous links are flagged, not discarded, in
the primary analysis"). Não há uma versão "sem desempate" observável — todo
vínculo, ambíguo ou não, já sai do pipeline com um vencedor. O que dá para
medir é o INVERSO: quanto do Cenário 0 depende de confiar nesse desempate.
Aqui, óbitos atribuídos pela Camada 2 (`obitos_scored.parquet`,
`link_obito_status == 'obito_SIM_C2'`) cuja chave era disputada por mais de
um paciente candidato (`link_obito_chave_ambigua == True`) são recodificados
para censura — tratados como potencialmente do paciente errado. **A leitura
correta da tabela é 2→0, não 0→2**: o Cenário 0 é o que a contaminação por
ambiguidade JÁ inclui; o Cenário 2 é o piso conservador sem ela. A diferença
entre os dois é a mesma fração já documentada na Seção 6 (18,8% da Camada 2
vem de chave ambígua).

**Cenário 3 (erro de grão)**: já medido e publicado em
`experimento/tabela_grao_sensibilidade.csv` (contar por registro de tumor
RHC em vez de por paciente) — só é lido e formatado aqui, não recomputado.

Para os Cenários 0-2 (mesma coorte, mesmo grão), ajusta-se também um Cox
univariado em `cat_estadiamento` (sem penalização: o
`penalizer=0.1` do modelo completo encolhe HR e muda a leitura do
gradiente de estádio), reportando o HR do estádio IV vs I como métrica
única de "a conclusão clínica muda entre cenários".

Uso:
    python cenarios_degradacao_linkage.py

Saídas em `experimento/`:
    tabela_cenarios_degradacao.csv
    fig_cenarios_degradacao_km.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

import gerar_relatorio_wilms as G

AQUI = Path(__file__).resolve().parent
EXPERIMENTO = AQUI / "experimento"
EXPERIMENTO.mkdir(exist_ok=True)


def _hr_estadio_iv(d: pd.DataFrame, T_col: str, E_col: str) -> tuple[float, float, float, int]:
    """Cox univariado em cat_estadiamento, sem penalização. Devolve
    (HR, ic_lo, ic_hi, n) do estádio IV vs I; NaN se não houver dado
    suficiente (mesmo padrão de guarda de `os_com_ic`)."""
    sub = d[d["cat_estadiamento"].isin(G.ESTADIOS)][[T_col, E_col, "cat_estadiamento"]].dropna()
    sub = sub[sub[T_col] > 0]
    if len(sub) < 30 or sub[E_col].sum() < 5:
        return (np.nan, np.nan, np.nan, len(sub))
    sub["cat_estadiamento"] = pd.Categorical(sub["cat_estadiamento"], categories=G.ESTADIOS)
    dum = pd.get_dummies(sub, columns=["cat_estadiamento"], drop_first=True)
    dum = dum.loc[:, (dum.nunique() > 1) | dum.columns.isin([T_col, E_col])]
    col_iv = [c for c in dum.columns if "IV" in c]
    if not col_iv:
        return (np.nan, np.nan, np.nan, len(sub))
    cph = CoxPHFitter()  # sem penalizer — ver docstring do módulo
    cph.fit(dum, duration_col=T_col, event_col=E_col)
    s = cph.summary
    row = s.loc[col_iv[0]]
    return (float(np.exp(row["coef"])), float(np.exp(row["coef lower 95%"])),
            float(np.exp(row["coef upper 95%"])), len(sub))


def construir_tabela() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    d, funil = G.carregar_coorte(colapsar=True)

    obitos = pd.read_parquet(AQUI / "obitos_scored.parquet",
                              columns=["id_unico", "link_obito_status", "link_obito_chave_ambigua"])
    d = d.merge(obitos, left_on="id_paciente", right_on="id_unico", how="left")

    cenarios: dict[str, pd.DataFrame] = {}

    # Cenário 0 — referência
    c0 = d[["T_meses", "E_final"]].copy()
    cenarios["0_referencia"] = c0

    # Cenário 1 — só camada C1 (intra-sistema), sem SIM e sem DATAOBITO do RHC.
    # O teste de horizonte é `E_final == 1`, não `T_meses <= TAU`: `T_meses` já
    # vem capado em TAU (gerar_relatorio_wilms), então aquela condição seria sempre
    # verdadeira e contaria como evento aos 60 meses os 8 óbitos C1 que ocorrem
    # DEPOIS dos 5 anos (260 óbitos em vez de 252).
    c1 = d[["T_meses"]].copy()
    c1["E_final"] = np.where(
        (d["link_obito_status"] == "obito_interno_C1") & (d["E_final"] == 1), 1, 0)
    assert not ((c1["E_final"] == 1) & (d["E_final"] == 0)).any(), \
        "Cenário 1 tem de ser subconjunto dos óbitos do Cenário 0"
    cenarios["1_claims_ingenuo"] = c1

    # Cenário 1b — sem linkage ao SIM, mas mantendo o DATAOBITO do RHC: é o
    # ponto 100% da varredura de perda de vínculo (propagacao_erro_linkage.py).
    c1b = d[["T_meses"]].copy()
    sim_dependente = (d["link_obito_status"] == "obito_SIM_C2") & (d["evento_obito_inca"] != 1)
    c1b["E_final"] = np.where(sim_dependente, 0, d["E_final"])
    cenarios["1b_sem_link_SIM"] = c1b

    # Cenário 2 — óbitos de chave ambígua na Camada 2 recodificados p/ censura
    ambiguo_c2 = (d["link_obito_status"] == "obito_SIM_C2") & (d["link_obito_chave_ambigua"] == True)  # noqa: E712
    c2 = d[["T_meses"]].copy()
    c2["E_final"] = np.where(ambiguo_c2, 0, d["E_final"])
    cenarios["2_ambiguidade_removida"] = c2

    linhas = []
    for nome, cen in cenarios.items():
        T, E = cen["T_meses"], cen["E_final"]
        n, eventos = len(cen), int(E.sum())
        s60, lo, hi = G.os_com_ic(T, E, G.TAU)
        # HR precisa das colunas de estádio — junta de volta de `d`
        cen_full = d[["cat_estadiamento"]].copy()
        cen_full["T_meses"], cen_full["E_final"] = T.values, E.values
        hr, hr_lo, hr_hi, n_cox = _hr_estadio_iv(cen_full, "T_meses", "E_final")
        linhas.append({
            "cenario": nome, "n": n, "obitos": eventos,
            "os_5a_pct": round(s60, 1) if pd.notna(s60) else np.nan,
            "os_5a_ic_lo": round(lo, 1) if pd.notna(lo) else np.nan,
            "os_5a_ic_hi": round(hi, 1) if pd.notna(hi) else np.nan,
            "hr_estadio_iv_vs_i": round(hr, 2) if pd.notna(hr) else np.nan,
            "hr_ic_lo": round(hr_lo, 2) if pd.notna(hr_lo) else np.nan,
            "hr_ic_hi": round(hr_hi, 2) if pd.notna(hr_hi) else np.nan,
            "n_cox": n_cox,
        })

    # Cenário 3 — reusa tabela já publicada (não recomputa)
    grao_path = EXPERIMENTO / "tabela_grao_sensibilidade.csv"
    if grao_path.exists():
        grao = pd.read_csv(grao_path).set_index("medida")
        linhas.append({
            "cenario": "3_grao_registro", "n": int(grao.loc["N", "por registro RHC"]),
            "obitos": int(grao.loc["óbitos", "por registro RHC"]),
            "os_5a_pct": round(grao.loc["OS 5 anos (%)", "por registro RHC"], 1),
            "os_5a_ic_lo": np.nan, "os_5a_ic_hi": np.nan,
            "hr_estadio_iv_vs_i": np.nan, "hr_ic_lo": np.nan, "hr_ic_hi": np.nan,
            "n_cox": np.nan,
        })
    else:
        print(f"⚠ {grao_path} não encontrado — Cenário 3 omitido "
              f"(rode analises_complementares.py/gerar_tabelas_suplementares.py antes)")

    tabela = pd.DataFrame(linhas)
    return tabela, cenarios


def figura_km(cenarios: dict[str, pd.DataFrame], destino: Path, idioma: str = "pt") -> None:
    """`idioma='en'` gera a versão com rótulos em inglês (para o manuscrito
    ML4H) sem alterar a saída em português já usada no projeto."""
    fig, ax = plt.subplots(figsize=(7, 5))
    rotulos_pt = {
        "0_referencia": "Cenário 0 — Referência (pipeline completo)",
        "1_claims_ingenuo": "Cenário 1 — Só SIH (sem SIM)",
        "1b_sem_link_SIM": "Cenário 1b — Sem link ao SIM (SIH + RHC)",
        "2_ambiguidade_removida": "Cenário 2 — Óbitos de chave ambígua removidos (censurados)",
    }
    rotulos_en = {
        "0_referencia": "Scenario 0 — Reference (full pipeline)",
        "1_claims_ingenuo": "Scenario 1 — Claims-only (SIH, no SIM link)",
        "1b_sem_link_SIM": "Scenario 1b — No SIM link (SIH + registry)",
        "2_ambiguidade_removida": "Scenario 2 — Ambiguous-key deaths removed",
    }
    rotulos = rotulos_en if idioma == "en" else rotulos_pt
    for nome, cen in cenarios.items():
        kmf = KaplanMeierFitter()
        kmf.fit(cen["T_meses"], cen["E_final"], label=rotulos.get(nome, nome))
        kmf.plot_survival_function(ax=ax, ci_show=False)
    if idioma == "en":
        ax.set_xlabel("Months since diagnosis")
        ax.set_ylabel("Overall survival")
        ax.set_title("Overall survival under the 3 linkage-degradation scenarios\n(Nephroblastoma, DATASUS 2008–2023)")
    else:
        ax.set_xlabel("Meses desde o diagnóstico")
        ax.set_ylabel("Sobrevida global")
        ax.set_title("Sobrevida global sob os 3 cenários de degradação de linkage\n(Nefroblastoma, DATASUS 2008–2023)")
    ax.set_xlim(0, G.TAU)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    if destino.suffix == ".png":
        fig.savefig(destino.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    tabela, cenarios = construir_tabela()
    tabela.to_csv(EXPERIMENTO / "tabela_cenarios_degradacao.csv", index=False)
    print(tabela.to_string(index=False))
    figura_km(cenarios, EXPERIMENTO / "fig_cenarios_degradacao_km.png")
    figura_km(cenarios, EXPERIMENTO / "fig_cenarios_degradacao_km_en.png", idioma="en")
    agregados = AQUI.parents[1] / "outputs"
    agregados.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(agregados / "tabela_cenarios_degradacao.csv", index=False)
    print(f"\n📄 {EXPERIMENTO}/tabela_cenarios_degradacao.csv")
    print(f"🖼  {EXPERIMENTO}/fig_cenarios_degradacao_km.png")


if __name__ == "__main__":
    main()
