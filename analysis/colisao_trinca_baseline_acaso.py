"""Linha de base de acaso para os 50,96% de trincas com mais de um CNS:
a taxa observada comparada com a esperada por acaso.

O paper publica que 50,96% das trincas (nascimento, sexo, município) observadas
no BI aparecem sob mais de um CNS válido. Pessoas diferentes compartilham trinca
por acaso — numa cidade grande, várias crianças do mesmo sexo nascem no mesmo dia —,
então o número só diz algo comparado com o que o acaso produz.

Modelo nulo: cada tupla (CNS, trinca) é uma pessoa distinta; dentro de cada célula
(município, sexo) com N tuplas, a data de nascimento é sorteada da distribuição
observada de datas da UF daquele município. O número de pessoas numa trinca é então
~Poisson(λ = N · p(data | UF)), e

    P(trinca ocupada)     = 1 − e^(−λ)
    P(≥ 2 pessoas)        = 1 − e^(−λ) − λ·e^(−λ)
    esperado(≥2 | ocupada) = Σ P(≥2) / Σ P(ocupada)   (somando células × datas)

Usar a distribuição de datas da UF (e não a nacional) incorpora a estrutura etária
regional; dentro da UF o município herda a dela, o que é uma aproximação declarada.
Se o observado ficar perto do esperado, os 50,96% são em grande parte colisão
legítima entre pessoas, não contaminação do identificador.

Lê o cache `trincas_shuffle_ate2025` de `validar_cns_bi.py` (Etapa B), partição a
partição, com a validade do CNS pela macro SQL `cns_dv_valido` (mesma conta de
`digito_verificador_valido`). Uma partição por vez, 3 GB.

Saídas em dados/stats/faltantes_cns_bi/ (+ outputs/):
    colisao_trinca_baseline_ate2025.csv
"""
from __future__ import annotations

import shutil

import duckdb
import numpy as np
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths
from aux_cns import registrar_macros

TRINCAS = _paths.CACHE / "faltantes_cns_bi" / "trincas_shuffle_ate2025"
OUT = _paths.STATS / "faltantes_cns_bi"
PARC = OUT / "_parcial_baseline_ate2025"
AGREGADOS = _paths.STATS.parents[1] / "outputs"


def particao(k: int) -> None:
    alvo = {n: PARC / f"{n}_part={k}.parquet" for n in ("celula", "data", "obs")}
    if all(p.exists() for p in alvo.values()):
        return
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=3")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{_paths.CACHE / 'tmp_duck' / f'baseline_p{k}'}'")
    registrar_macros(con)
    con.execute(f"""CREATE TEMP TABLE t AS
        SELECT DISTINCT cns, nasc, sexo, mun_res
        FROM read_parquet('{(TRINCAS / f"part={k}" / "*.parquet").as_posix()}')
        WHERE cns_dv_valido(trim(cns))""")
    con.execute(f"""COPY (SELECT mun_res, sexo, COUNT(*) AS n FROM t GROUP BY ALL)
                    TO '{alvo["celula"]}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT substr(mun_res, 1, 2) AS uf, nasc, COUNT(*) AS n FROM t GROUP BY ALL)
                    TO '{alvo["data"]}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT n_cns, COUNT(*) AS n_trincas FROM
                          (SELECT nasc, sexo, mun_res, COUNT(DISTINCT cns) AS n_cns FROM t GROUP BY ALL)
                          GROUP BY n_cns) TO '{alvo["obs"]}' (FORMAT parquet)""")
    print(f"  partição {k} ok", flush=True)


def main() -> None:
    PARC.mkdir(parents=True, exist_ok=True)
    partes = sorted(int(p.name.split("=")[1]) for p in TRINCAS.glob("part=*"))
    for k in partes:
        particao(k)
    con = duckdb.connect()
    obs = con.execute(f"SELECT n_cns, SUM(n_trincas) n FROM '{PARC}/obs_part=*.parquet' GROUP BY 1").df()
    ocup_obs, multi_obs = obs.n.sum(), obs.loc[obs.n_cns > 1, "n"].sum()
    pub = pd.read_csv(OUT / "multiplicidade_teto_ate2025.csv")
    assert ocup_obs == pub.n_trincas.sum() and multi_obs == pub.loc[pub.n_cns_distintos_na_trinca > 1, "n_trincas"].sum(), \
        (ocup_obs, multi_obs, "não reproduz multiplicidade_teto_ate2025.csv")

    cel = con.execute(f"SELECT mun_res, sexo, SUM(n) n FROM '{PARC}/celula_part=*.parquet' GROUP BY ALL").df()
    dat = con.execute(f"SELECT uf, nasc, SUM(n) n FROM '{PARC}/data_part=*.parquet' GROUP BY ALL").df()
    cel["uf"] = cel.mun_res.str[:2]
    esp_ocup = esp_multi = 0.0
    linhas_uf = []
    for uf, g in cel.groupby("uf"):
        p = dat.loc[dat.uf == uf, "n"].to_numpy(float)
        p /= p.sum()
        o = m = 0.0
        for N in g.n.to_numpy(float):  # uma célula por vez: vetor de datas da UF
            lam = N * p
            e = np.exp(-lam)
            o += (1 - e).sum()
            m += (1 - e - lam * e).sum()
        esp_ocup += o
        esp_multi += m
        linhas_uf.append({"uf": uf, "esperado_pct_multi": 100 * m / o})
    t = pd.DataFrame([{
        "trincas_observadas": int(ocup_obs), "trincas_multi_obs": int(multi_obs),
        "pct_multi_obs": 100 * multi_obs / ocup_obs,
        "trincas_esperadas_acaso": esp_ocup, "trincas_multi_esperadas_acaso": esp_multi,
        "pct_multi_esperado_acaso": 100 * esp_multi / esp_ocup,
        "razao_obs_esperado": (multi_obs / ocup_obs) / (esp_multi / esp_ocup),
        "excesso_trincas_multi": multi_obs - esp_multi * ocup_obs / esp_ocup,
        "tuplas": int(cel.n.sum()), "celulas_mun_sexo": len(cel),
        "nota": "nulo Poisson por célula (município×sexo), datas pela distribuição da UF",
    }])
    print(t.T.to_string())
    t.to_csv(OUT / "colisao_trinca_baseline_ate2025.csv", index=False)
    pd.DataFrame(linhas_uf).to_csv(OUT / "colisao_trinca_baseline_por_uf_ate2025.csv", index=False)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "colisao_trinca_baseline_ate2025.csv", AGREGADOS / "colisao_trinca_baseline_ate2025.csv")
    print(f"✅ {OUT / 'colisao_trinca_baseline_ate2025.csv'}")


if __name__ == "__main__":
    main()
