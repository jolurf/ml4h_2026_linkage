"""Por que o SIM acha só 42% dos óbitos do RHC no C50.

No modelo do C50, 18 144 óbitos existem só no RHC e não têm gradiente de estádio
(`c50_obitos_por_fonte_estadio.py`). Duas explicações, com consequências opostas:
o link com o SIM falhou (o óbito existe e o rótulo SIM está incompleto) ou o link
RHC↔SUS ligou o registro de óbito à pessoa errada (o óbito não é dela).

A busca dirigida do nefroblastoma (`captura_recaptura_obitos.py`) é refeita aqui:
cada óbito só-RHC é procurado no SIM pela chave data do óbito + data de
nascimento, sem município e sem CID. A linha de base é o mesmo teste nos óbitos
que as duas fontes têm. Por estádio, para ver se o que não se acha no SIM é o
que não tem gradiente.

Lincoln-Petersen com as duas listas (SIM, RHC) sai ao lado só como contraste:
com vínculos falsos na lista do RHC ele superestima o total.

Mesma população de `c50_calibracao_rotulos.carregar` (209 387 pacientes).
Saída: projetos/C50/resultados/decomposicao_so_rhc_c50.csv (+ outputs/)
"""
from __future__ import annotations

import shutil

import duckdb
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths
from c50_calibracao_rotulos import BASE, OUT, AGREGADOS, carregar

DO = _paths.DADOS / "parquets" / "DO" if hasattr(_paths, "DADOS") else _paths.STATS.parent / "parquets" / "DO"


def main() -> None:
    d = carregar()
    nasc = pd.read_parquet(BASE / "intermediarios" / "coorte_sobrevida.parquet",
                           columns=["id_unico", "data_nascimento"])
    d = d.merge(nasc, on="id_unico", how="left", validate="one_to_one")
    d["grupo"] = (d.E_SIM.astype(str) + d.E_RHC.astype(str)).map(
        {"10": "so_sim", "01": "so_rhc", "11": "ambos", "00": "vivo"})
    alvo = d[d.grupo.isin(["so_rhc", "ambos"])].copy()
    alvo["dtobito"] = pd.to_datetime(alvo["dt_obito_rhc"]).dt.strftime("%d%m%Y")
    alvo["dtnasc"] = pd.to_datetime(alvo["data_nascimento"]).dt.strftime("%d%m%Y")
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=3")
    con.execute(f"SET temp_directory='{_paths.CACHE / 'tmp_duck' / 'c50_so_rhc'}'")
    con.register("a", alvo[["id_unico", "dtobito", "dtnasc"]])
    achados = con.execute(f"""
        SELECT a.id_unico, COUNT(*) AS n_cand
        FROM a JOIN read_parquet('{DO}/**/*.parquet', hive_partitioning=true, union_by_name=true) s
          ON s.DTOBITO = a.dtobito AND s.DTNASC = a.dtnasc
        GROUP BY a.id_unico""").df()
    alvo = alvo.merge(achados, on="id_unico", how="left")
    alvo["achou"] = alvo["n_cand"].notna()
    linhas = []
    for g, sub in alvo.groupby("grupo"):
        linhas.append({"grupo": g, "estadio": "todos", "n": len(sub), "achados_sim": int(sub.achou.sum()),
                       "pct_achados_sim": 100 * sub.achou.mean()})
        for st, ss in sub.groupby("staging"):
            linhas.append({"grupo": g, "estadio": st, "n": len(ss), "achados_sim": int(ss.achou.sum()),
                           "pct_achados_sim": 100 * ss.achou.mean()})
    t = pd.DataFrame(linhas)
    base = t[(t.grupo == "ambos") & (t.estadio == "todos")].iloc[0].pct_achados_sim / 100
    so = t[(t.grupo == "so_rhc") & (t.estadio == "todos")].iloc[0]
    # verdadeiros = achados / taxa de acerto da chave; o resto são prováveis falsos.
    # Limite superior: a taxa de acerto foi medida em óbitos que o link já achava.
    falsos = max(0.0, so.n - so.achados_sim / base)
    n_sim, n_rhc = int((d.E_SIM == 1).sum()), int((d.E_RHC == 1).sum())
    m = int(((d.E_SIM == 1) & (d.E_RHC == 1)).sum())
    lp = n_sim * n_rhc / m
    resumo = pd.DataFrame([
        {"grupo": "RESUMO", "estadio": "taxa de não-encontrados entre os 'ambos' (linha de base)", "n": None,
         "achados_sim": None, "pct_achados_sim": 100 * (1 - base)},
        {"grupo": "RESUMO", "estadio": "só-RHC não encontrados além da linha de base (prováveis falsos)",
         "n": round(falsos), "achados_sim": None, "pct_achados_sim": 100 * falsos / so.n},
        {"grupo": "RESUMO", "estadio": "Lincoln-Petersen SIM × RHC (contraste; inflado por vínculos falsos)",
         "n": round(lp), "achados_sim": None, "pct_achados_sim": 100 * lp / len(d)},
    ])
    t = pd.concat([t, resumo], ignore_index=True)
    print(t.round(1).to_string(index=False))
    t.to_csv(OUT / "decomposicao_so_rhc_c50.csv", index=False)
    shutil.copy2(OUT / "decomposicao_so_rhc_c50.csv", AGREGADOS / "decomposicao_so_rhc_c50.csv")
    print(f"✅ {OUT / 'decomposicao_so_rhc_c50.csv'}")


if __name__ == "__main__":
    main()
