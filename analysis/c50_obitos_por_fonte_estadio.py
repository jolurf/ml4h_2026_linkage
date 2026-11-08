"""Óbitos do C50 por fonte e estádio: por que o HR da União fica entre SIM e RHC.

A União marca óbito se SIM **ou** RHC o registram, com a data mais antiga; tem
mais óbitos que qualquer fonte, mas o HR é uma razão entre estádios, não uma
contagem. Ele cai se os óbitos que a União acrescenta não dependem do estádio.
Aqui: a fração de pacientes em cada grupo de fonte (só SIM, só RHC, ambos) por
estádio, e a mortalidade por estádio sob cada rótulo, com a razão IV/I.

Mesma população de `c50_calibracao_rotulos.carregar` (209 387 pacientes).
Saída: projetos/C50/resultados/obitos_por_fonte_estadio.csv (+ outputs/)
"""
from __future__ import annotations

import shutil

import duckdb
import pandas as pd

import _raiz  # noqa: F401
from c50_calibracao_rotulos import BASE, OUT, AGREGADOS, carregar


def main() -> None:
    d = carregar()
    grupo = (d.E_SIM.astype(str) + d.E_RHC.astype(str)).map(
        {"10": "so_SIM", "01": "so_RHC", "11": "ambos", "00": "vivo"})
    linhas = []
    for st, g in d.assign(grupo=grupo).groupby("staging"):
        linha = {"estadio": st, "n": len(g)}
        for k in ("so_SIM", "so_RHC", "ambos"):
            linha[f"pct_{k}"] = 100 * (g.grupo == k).mean()
        for rot in ("SIM", "RHC", "Uniao"):
            linha[f"mortalidade_{rot}"] = 100 * g[f"E_{rot}"].mean()
        linhas.append(linha)
    t = pd.DataFrame(linhas)
    i, iv = t.set_index("estadio").loc["I"], t.set_index("estadio").loc["IV"]
    razao = {"estadio": "razão IV/I", "n": None,
             **{f"mortalidade_{r}": iv[f"mortalidade_{r}"] / i[f"mortalidade_{r}"] for r in ("SIM", "RHC", "Uniao")},
             **{f"pct_{k}": iv[f"pct_{k}"] / i[f"pct_{k}"] for k in ("so_SIM", "so_RHC", "ambos")}}
    t = pd.concat([t, pd.DataFrame([razao])], ignore_index=True)
    n_grupo = grupo.value_counts()
    t.attrs["n"] = n_grupo.to_dict()
    for k in ("so_SIM", "so_RHC", "ambos"):
        t[f"n_total_{k}"] = int(n_grupo.get(k, 0))
    print(t.round(2).to_string(index=False))
    t.to_csv(OUT / "obitos_por_fonte_estadio.csv", index=False)
    shutil.copy2(OUT / "obitos_por_fonte_estadio.csv", AGREGADOS / "obitos_por_fonte_estadio.csv")
    print(f"✅ {OUT / 'obitos_por_fonte_estadio.csv'}")

    # O rótulo "SIM" é o do pipeline: C1 = flag de óbito no próprio registro (AQ/AR/SIH,
    # sem link, datado na última aparição) + C2 = link demográfico ao SIM. Por camada,
    # confirmação pelo RHC e mortalidade por estádio.
    con = duckdb.connect()
    con.register("d", d[["id_unico", "E_SIM", "E_RHC", "staging"]])
    cam = con.execute(f"""
        SELECT d.staging AS estadio, COUNT(*) AS n,
               SUM(COALESCE(c.camada_obito = 1 AND d.E_SIM = 1, false)::INT) AS obitos_c1,
               SUM(COALESCE(c.camada_obito = 2 AND d.E_SIM = 1, false)::INT) AS obitos_c2,
               SUM(COALESCE(c.camada_obito = 1 AND d.E_SIM = 1 AND d.E_RHC = 1, false)::INT) AS c1_confirmados_rhc,
               SUM(COALESCE(c.camada_obito = 2 AND d.E_SIM = 1 AND d.E_RHC = 1, false)::INT) AS c2_confirmados_rhc
        FROM d JOIN '{BASE}/intermediarios/coorte_sobrevida.parquet' c USING (id_unico)
        GROUP BY ROLLUP (d.staging) ORDER BY d.staging NULLS LAST""").df()
    cam["estadio"] = cam["estadio"].fillna("todos")
    cam["mortalidade_c1"] = 100 * cam.obitos_c1 / cam.n
    cam["mortalidade_c2"] = 100 * cam.obitos_c2 / cam.n
    print(cam.round(2).to_string(index=False))
    cam.to_csv(OUT / "obitos_pipeline_por_camada.csv", index=False)
    shutil.copy2(OUT / "obitos_pipeline_por_camada.csv", AGREGADOS / "obitos_pipeline_por_camada.csv")
    print(f"✅ {OUT / 'obitos_pipeline_por_camada.csv'}")


if __name__ == "__main__":
    main()
