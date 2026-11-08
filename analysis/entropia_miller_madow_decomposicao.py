"""Entropia cadastral: correção de viés e decomposição por campo.

Dois refinamentos da entropia de `entropia_cadastral_cns.py`:

1. **Viés do estimador plug-in.** Com poucos registros por CNS a entropia
   plug-in subestima a verdadeira. Aplica-se Miller–Madow por CNS,
   H_MM = H + (K − 1) / (2 N ln 2) em bits (K identidades observadas, N
   registros), sobre os parciais por CNS já gravados — sem reprocessar o BI.
   A fração de CNS com mais de uma identidade não muda (é contagem, não
   estimativa); muda a magnitude média.
2. **Mudança legítima × erro.** A identidade é a trinca nascimento|sexo|município.
   Trocar de município é mudança de endereço, e não erro de cadastro; trocar de
   nascimento ou sexo é contaminação do identificador. `--decompor` classifica cada
   CNS com mais de uma identidade pelo(s) campo(s) que variam (uma passada pelo
   cache `cascade_bi_estabilidade_ate2025`, uma partição por subprocesso).
   Componente nulo conta como valor próprio, como no script original (a trinca
   nula é uma identidade a mais).

Saídas em dados/stats/entropia_cadastral_cns/ (+ outputs/):
    entropia_miller_madow_ate2025.csv
    decomposicao_identidades_ate2025.csv      (com --decompor)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import duckdb
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths
from cascade_bi_estabilidade import _cache_dir as _cascade_cache_dir

OUT = _paths.STATS / "entropia_cadastral_cns"
PARC = OUT / "_parcial_ate2025"
DEC = OUT / "_parcial_decomposicao_ate2025"
AGREGADOS = _paths.STATS.parents[1] / "outputs"
FAIXA = """CASE WHEN n_total = 1 THEN '1' WHEN n_total <= 4 THEN '2-4' WHEN n_total <= 9 THEN '5-9'
               WHEN n_total <= 24 THEN '10-24' WHEN n_total <= 49 THEN '25-49'
               WHEN n_total <= 99 THEN '50-99' ELSE '100+' END"""
ORDEM = {"1": 0, "2-4": 1, "5-9": 2, "10-24": 3, "25-49": 4, "50-99": 5, "100+": 6, "total": 7}


def _con(tag: str = "main") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=3")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{_paths.CACHE / 'tmp_duck' / ('entropia_' + tag)}'")
    return con


def gravar(df: pd.DataFrame, nome: str) -> None:
    df.to_csv(OUT / nome, index=False)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / nome, AGREGADOS / nome)
    print(f"✅ {OUT / nome}")


def miller_madow() -> None:
    con = _con()
    sel = f"""SELECT {FAIXA} AS faixa, n_total, n_identidades, entropia_bits,
                 entropia_bits + (n_identidades - 1) / (2 * n_total * ln(2)) AS entropia_mm
          FROM read_parquet('{PARC}/*.parquet')"""
    agg = """COUNT(*) AS n_cns, AVG(entropia_bits) AS entropia_plugin_media,
             AVG(entropia_mm) AS entropia_mm_media,
             100.0 * AVG((n_identidades > 1)::INT) AS pct_multi_identidade,
             AVG(entropia_bits) FILTER (WHERE n_identidades > 1) AS plugin_media_multi,
             AVG(entropia_mm) FILTER (WHERE n_identidades > 1) AS mm_media_multi"""
    t = con.execute(f"""SELECT faixa, {agg} FROM ({sel}) GROUP BY faixa
                        UNION ALL SELECT 'total', {agg} FROM ({sel})""").df()
    t = t.sort_values("faixa", key=lambda s: s.map(ORDEM)).reset_index(drop=True)
    tot = t.set_index("faixa").loc["total"]
    assert int(tot.n_cns) == 235_645_653 and abs(tot.pct_multi_identidade - 19.81) < 0.01, tot
    print(t.round(4).to_string(index=False))
    gravar(t, "entropia_miller_madow_ate2025.csv")


def _particao(k: int) -> None:
    con = _con(f"p{k}")
    pasta = _cascade_cache_dir(2025) / f"part={k}"
    DEC.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (
        WITH b AS (SELECT cns, COALESCE(dtnasc, '∅') AS dt, COALESCE(sexo, '∅') AS sx,
                          COALESCE(mun_res, '∅') AS mu
                   FROM read_parquet('{pasta.as_posix()}/*.parquet') WHERE cns_status = 'valido_candidato'),
             c AS (SELECT cns, COUNT(DISTINCT dt) AS k_dt, COUNT(DISTINCT sx) AS k_sx,
                          COUNT(DISTINCT mu) AS k_mu, COUNT(DISTINCT (dt, sx, mu)) AS k_id
                   FROM b GROUP BY cns)
        SELECT (k_dt > 1) AS varia_nasc, (k_sx > 1) AS varia_sexo, (k_mu > 1) AS varia_mun,
               COUNT(*) AS n_cns
        FROM c WHERE k_id > 1 GROUP BY ALL
    ) TO '{(DEC / f"part={k}.parquet.tmp").as_posix()}' (FORMAT parquet)""")
    (DEC / f"part={k}.parquet.tmp").rename(DEC / f"part={k}.parquet")  # só existe se o COPY terminou
    print(f"  decomposição: partição {k} ok", flush=True)


def decompor(paralelo: int) -> None:
    pend = [k for k in range(24) if not (DEC / f"part={k}.parquet").exists()]
    ativos: list[subprocess.Popen] = []
    for k in pend:
        while len(ativos) >= paralelo:
            ativos[0].wait()
            for p in [p for p in ativos if p.poll() is not None]:
                assert p.returncode == 0, "partição falhou"
                ativos.remove(p)
        ativos.append(subprocess.Popen([sys.executable, __file__, "--_particao", str(k)]))
    for p in ativos:
        p.wait()
        assert p.returncode == 0, "partição falhou"
    con = _con()
    t = con.execute(f"""SELECT varia_nasc, varia_sexo, varia_mun, SUM(n_cns) AS n_cns
                        FROM read_parquet('{DEC}/*.parquet') GROUP BY ALL""").df()
    rot = {(False, False, True): "só município", (True, False, False): "só nascimento",
           (False, True, False): "só sexo", (True, True, False): "nascimento e sexo",
           (True, False, True): "nascimento e município", (False, True, True): "sexo e município",
           (True, True, True): "os três"}
    t["categoria"] = [rot[(a, b, c)] for a, b, c in zip(t.varia_nasc, t.varia_sexo, t.varia_mun)]
    t["pct_dos_multi"] = 100 * t.n_cns / t.n_cns.sum()
    t = t.sort_values("n_cns", ascending=False)
    n_multi = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARC}/*.parquet') WHERE n_identidades > 1").fetchone()[0]
    assert t.n_cns.sum() == n_multi, (t.n_cns.sum(), n_multi)
    resumo = pd.DataFrame([
        {"categoria": "RESUMO: só município varia (mudança de endereço plausível)",
         "n_cns": int(t.loc[t.categoria == "só município", "n_cns"].sum())},
        {"categoria": "RESUMO: nascimento ou sexo varia (contaminação do identificador)",
         "n_cns": int(t.loc[t.varia_nasc | t.varia_sexo, "n_cns"].sum())}])
    resumo["pct_dos_multi"] = 100 * resumo.n_cns / n_multi
    t = pd.concat([t, resumo], ignore_index=True)
    print(t.round(2).to_string(index=False))
    gravar(t, "decomposicao_identidades_ate2025.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decompor", action="store_true")
    ap.add_argument("--paralelo", type=int, default=3)
    ap.add_argument("--_particao", type=int)
    a = ap.parse_args()
    if a._particao is not None:
        return _particao(a._particao)
    miller_madow()
    if a.decompor:
        decompor(a.paralelo)


if __name__ == "__main__":
    main()
