#!/usr/bin/env python3
"""Validação do join RHC↔SUS com o CNS como gabarito.

O RHC não tem CNS, então o join que ancora a coorte (`2_get_cancer_histo`,
célula 7) é uma cascata demográfica: bloco por CNES do hospital + sexo, depois
município de residência + sexo, com ±1 ano de nascimento e ±75 dias entre as
datas de diagnóstico, escolhendo o candidato de menor diferença de dias. Não dá
para medir o erro sobre o RHC. Dá para medir sobre o SIA, onde o CNS existe:

- **sonda** = cada registro AQ/AR (tem `data_identificacao`, a data de
  diagnóstico), no papel do registro do RHC;
- **referência** = uma pessoa por `id_unico` (CNES frequentados, município e ano
  de nascimento modais, 1ª data de diagnóstico), no papel do lado SUS;
- liga-se **cego** (sem o CNS) com a mesma cascata; depois o CNS diz acerto/erro.

Transcreve as células 19–23 (tag `exploratorio`) do notebook, que não rodam no
pipeline. Limitação a declarar: a sonda é um dos registros que formam a referência,
então data e CNES concordam em parte por construção — os números são **limite
superior** de precisão e recall.

Saídas em `data_saida/` (+ outputs/):
    validacao_rhc_roc.csv      sensibilidade/especificidade/precisão por bloco e janela
    validacao_rhc_cascata.csv  cascata operacional por janela: cobertura, acurácia, recall
    validacao_rhc_resumo.csv   a janela de produção (75 d): cobertura, precisão, FDR, recall
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "data_saida"
AGREGADOS = AQUI.parents[1] / "outputs"
JANELA_ANOS_NASC = 1
JANELA_MAX_DIAS = 360
PASSO_DIAS = 15
JANELA_PROD = 75


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.strip(), format="%Y%m%d", errors="coerce")


def fontes() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["subbase", "cns", "id_unico", "cnes_unidade", "mun_res_pac", "sexo",
            "data_nascimento", "data_identificacao"]
    d = pd.read_parquet(AQUI / "df_final_scored.parquet", columns=cols)
    d["dt_nasc"] = _to_date(d["data_nascimento"])
    d["cnes"] = d["cnes_unidade"].astype(str).str.split(".").str[0].str.zfill(7)
    d["mun"] = d["mun_res_pac"].astype(str).str.split(".").str[0].str.slice(0, 6)
    d["sexo"] = d["sexo"].astype(str).str.strip()

    probe = d[d["subbase"].isin(["AQ", "AR"])].copy()
    probe["dt_diag"] = _to_date(probe["data_identificacao"])
    probe = probe.dropna(subset=["dt_diag", "dt_nasc"])
    probe = probe[probe["sexo"].isin(["1", "2"])].reset_index(drop=True)
    probe["probe_id"] = probe.index
    probe["by"] = probe["dt_nasc"].dt.year
    P = probe[["probe_id", "cns", "sexo", "by", "mun", "cnes", "dt_diag"]].rename(columns={"cns": "cns_p"})

    ref_diag = probe.groupby("id_unico")["dt_diag"].min().rename("ref_dt_diag")
    # modas com desempate pelo menor valor (ordem total, determinística)
    def moda(x: pd.Series):
        x = x.dropna()
        return None if x.empty else x.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]
    persons = d.groupby("id_unico").agg(
        cns=("cns", "first"), sexo=("sexo", moda), mun=("mun", moda),
        by=("dt_nasc", lambda x: moda(x.dropna().dt.year))).join(ref_diag)
    persons = persons.dropna(subset=["ref_dt_diag", "by"]).reset_index()
    persons["by"] = persons["by"].astype(int)
    cnes_set = (d.dropna(subset=["cnes"]).groupby("id_unico")["cnes"]
                .apply(lambda s: sorted(set(s))).rename("cnes_set").reset_index())
    R = persons.merge(cnes_set, on="id_unico", how="left")
    return P, R


def candidatos(P: pd.DataFrame, R: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.register("P", P)
    con.register("Rp", R[["id_unico", "cns", "sexo", "by", "mun", "ref_dt_diag"]].rename(columns={"cns": "cns_r"}))
    con.register("Rc", R[["id_unico", "cnes_set"]].explode("cnes_set").rename(columns={"cnes_set": "cnes_r"}))
    base = """SELECT p.probe_id, p.cns_p, r.cns_r,
                     ABS(DATEDIFF('day', p.dt_diag, r.ref_dt_diag)) AS diff_dias,
                     ABS(p.by - r.by) AS diff_anos, (p.cns_p = r.cns_r) AS same"""
    cand_cnes = con.execute(f"""{base} FROM P p JOIN Rc rc ON p.cnes = rc.cnes_r JOIN Rp r ON r.id_unico = rc.id_unico
        WHERE p.sexo = r.sexo AND ABS(p.by - r.by) <= {JANELA_ANOS_NASC}""").df()
    cand_mun = con.execute(f"""{base} FROM P p JOIN Rp r ON p.mun = r.mun
        WHERE p.sexo = r.sexo AND ABS(p.by - r.by) <= {JANELA_ANOS_NASC}""").df()
    return cand_cnes, cand_mun


def curva_roc(cand: pd.DataFrame) -> pd.DataFrame:
    d, same = cand["diff_dias"].values, cand["same"].values.astype(bool)
    pos, neg = same.sum(), (~same).sum()
    linhas = []
    for w in range(0, JANELA_MAX_DIAS + 1, PASSO_DIAS):
        pred = d <= w
        tp, fp = int((pred & same).sum()), int((pred & ~same).sum())
        linhas.append(dict(janela=w, sensibilidade=tp / pos, especificidade=(neg - fp) / neg,
                           precisao=tp / (tp + fp) if tp + fp else np.nan, tp=tp, fp=fp))
    r = pd.DataFrame(linhas)
    r["youden_J"] = r.sensibilidade + r.especificidade - 1
    return r


def melhor(c: pd.DataFrame, w: int, excluir: set | None = None) -> pd.DataFrame:
    c = c[c["diff_dias"] <= w]
    if excluir:
        c = c[~c["probe_id"].isin(excluir)]
    # mesmo desempate do join de produção: diff_dias, diff_anos, e id do candidato
    return (c.sort_values(["probe_id", "diff_dias", "diff_anos", "cns_r"], kind="stable")
            .drop_duplicates("probe_id", keep="first"))


def main() -> None:
    P, R = fontes()
    n_true = int(P["cns_p"].isin(set(R["cns"])).sum())
    print(f"sondas AQ/AR: {len(P):,} | com CNS na referência: {n_true:,} | pessoas: {len(R):,}")
    cc, cm = candidatos(P, R)
    roc = pd.concat([curva_roc(cc).assign(bloco="CNES+sexo (R1)"),
                     curva_roc(cm).assign(bloco="Municipio+sexo (R3)")], ignore_index=True)
    linhas = []
    for w in range(0, JANELA_MAX_DIAS + 1, PASSO_DIAS):
        s1 = melhor(cc, w)
        s2 = melhor(cm, w, set(s1["probe_id"]))
        best = pd.concat([s1, s2])
        n, ac = len(best), int(best["same"].sum())
        linhas.append(dict(janela=w, cobertura=n / len(P), acuracia_cns=ac / n if n else np.nan,
                           recall=ac / n_true, n_vinculos=n,
                           n_cnes=len(s1), acertos_cnes=int(s1["same"].sum()),
                           n_mun=len(s2), acertos_mun=int(s2["same"].sum())))
    op = pd.DataFrame(linhas)
    r75 = op.set_index("janela").loc[JANELA_PROD]
    resumo = pd.DataFrame([{
        "janela_dias": JANELA_PROD, "sondas": len(P), "sondas_com_cns_na_referencia": n_true,
        "vinculos_entregues": int(r75.n_vinculos), "cobertura": r75.cobertura,
        "precisao": r75.acuracia_cns, "fdr": 1 - r75.acuracia_cns, "recall": r75.recall,
        "precisao_bloco_cnes": r75.acertos_cnes / r75.n_cnes, "fdr_bloco_cnes": 1 - r75.acertos_cnes / r75.n_cnes,
        "precisao_bloco_mun": r75.acertos_mun / r75.n_mun if r75.n_mun else np.nan,
        "fdr_bloco_mun": 1 - r75.acertos_mun / r75.n_mun if r75.n_mun else np.nan,
        "n_bloco_cnes": int(r75.n_cnes), "n_bloco_mun": int(r75.n_mun),
        "nota": "sonda é um dos registros da referência: limite superior de precisão e recall",
    }])
    for b in roc.bloco.unique():
        j = roc[roc.bloco == b].loc[lambda x: x.youden_J.idxmax()]
        resumo[f"youden_{b.split(' ')[0]}_janela"] = int(j.janela)
        resumo[f"youden_{b.split(' ')[0]}_fdr"] = 1 - j.precisao
    print(op[op.janela.isin([0, 30, 60, 75, 120, 180, 360])].round(4).to_string(index=False))
    print(resumo.T.to_string())
    SAIDA.mkdir(exist_ok=True)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    for df, nome in [(roc, "validacao_rhc_roc.csv"), (op, "validacao_rhc_cascata.csv"),
                     (resumo, "validacao_rhc_resumo.csv")]:
        df.to_csv(SAIDA / nome, index=False)
        shutil.copy2(SAIDA / nome, AGREGADOS / nome)
        print(f"✅ {SAIDA / nome}")


if __name__ == "__main__":
    main()
