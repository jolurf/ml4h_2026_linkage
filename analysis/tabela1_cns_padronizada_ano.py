"""Não-confiabilidade do CNS por raça/cor, padronizada pelo ano de atendimento.

O ano é o preditor mais forte de CNS não confiável (V de Cramér 0,324, contra
0,197 da raça) e a composição racial dos registros muda ao longo de 2008–2025.
Se as raças diferem só porque se concentram em anos diferentes, a padronização
direta pela distribuição anual TOTAL dos registros apaga a diferença.

Lê as células já agregadas por `tabela1_cns_nacional.py`
(`_parcial_ate2025/part=*.parquet`: sexo × raça × … × ano × confiável → registros),
sem nova varredura. Além do padronizado, publica a amplitude entre raças dentro
de cada ano.

Saídas em dados/stats/tabela1_cns_bi/ (+ outputs/):
    tabela1_raca_padronizada_ano_ate2025.csv
    tabela1_raca_por_ano_ate2025.csv
"""
from __future__ import annotations

import shutil

import duckdb
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths

DIR = _paths.STATS / "tabela1_cns_bi"
AGREGADOS = _paths.STATS.parents[1] / "outputs"


def main() -> None:
    c = duckdb.connect().execute(f"""
        SELECT raca_cor, ano, SUM(registros) AS n,
               SUM(registros) FILTER (WHERE NOT tem_cns_confiavel) AS n_nao_conf
        FROM read_parquet('{DIR}/_parcial_ate2025/*.parquet') GROUP BY ALL""").df()
    c["n_nao_conf"] = c["n_nao_conf"].fillna(0)
    c = c[c.raca_cor != "Não informado"]  # 2 registros
    peso_ano = c.groupby("ano").n.sum() / c.n.sum()
    c["taxa"] = c.n_nao_conf / c.n

    linhas = []
    for raca, g in c.groupby("raca_cor"):
        g = g.set_index("ano")
        bruta = 100 * g.n_nao_conf.sum() / g.n.sum()
        # anos sem registro da raça: renormaliza o peso sobre os anos presentes
        w = peso_ano.reindex(g.index)
        padr = 100 * (g.taxa * w).sum() / w.sum()
        linhas.append({"raca_cor": raca, "n_registros": int(g.n.sum()),
                       "pct_nao_confiavel_bruto": bruta, "pct_nao_confiavel_padronizado_ano": padr,
                       "anos_cobertos": len(g)})
    t = pd.DataFrame(linhas).sort_values("pct_nao_confiavel_bruto", ascending=False)
    pub = pd.read_csv(DIR / "tabela1_nacional_ate2025.csv", header=None)
    pub = pub[pub[0] == "raca_cor"].set_index(1)[5].astype(float)
    for _, r in t.iterrows():
        assert abs(r.pct_nao_confiavel_bruto - pub[r.raca_cor]) < 1e-6, (r.raca_cor, r.pct_nao_confiavel_bruto)
    sem_si = t[t.raca_cor != "Sem informação"]
    t.loc[len(t)] = {"raca_cor": "AMPLITUDE entre raças informadas (máx − mín)",
                     "pct_nao_confiavel_bruto": sem_si.pct_nao_confiavel_bruto.max() - sem_si.pct_nao_confiavel_bruto.min(),
                     "pct_nao_confiavel_padronizado_ano": sem_si.pct_nao_confiavel_padronizado_ano.max()
                     - sem_si.pct_nao_confiavel_padronizado_ano.min()}
    print(t.round(2).to_string(index=False))

    por_ano = (c.assign(pct=100 * c.taxa).pivot(index="ano", columns="raca_cor", values="pct"))
    por_ano["n_registros_ano"] = c.groupby("ano").n.sum()
    for d, nome in [(t, "tabela1_raca_padronizada_ano_ate2025.csv"),
                    (por_ano.reset_index(), "tabela1_raca_por_ano_ate2025.csv")]:
        d.to_csv(DIR / nome, index=False)
        AGREGADOS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DIR / nome, AGREGADOS / nome)
        print(f"✅ {DIR / nome}")


if __name__ == "__main__":
    main()
