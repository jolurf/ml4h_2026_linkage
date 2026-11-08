"""Recall E precisão da cascata R1→R3→R4 com o CNS escondido, dentro do BI.

`cascade_bi_recall_interno.py` dá o recall de cada chave sozinha, mas só olha
os registros do PRÓPRIO paciente: não há outro paciente no pool, então não há
como errar de pessoa e a precisão não é medida. Aqui o pool tem todo mundo.

Desenho (por UF inteira — amostrar pacientes ao acaso afinaria o pool de
não-pares e inflaria a precisão):
  - registros válidos do BI cujo município de residência está na UF;
  - de cada paciente com ≥2 registros, UM registro é retido (o de menor hash
    com sal) e vira a CONSULTA, no papel da AIH que chega sem CNS;
  - o POOL são todos os outros registros da UF, de todos os pacientes, no papel
    das identidades do SIA;
  - a consulta procura no pool pela R1 (nasc+CNES), depois R3 (nasc+sexo+mun_res),
    depois R4 (nasc+sexo+mun_ubs); a primeira chave que acha alguém decide, e o
    paciente escolhido é o MIN(cns) dos candidatos — o desempate da produção;
  - o CNS escondido é o gabarito.

Por regra sozinha e pela cascata acumulada:
  encontrado  = o CNS certo está entre os candidatos (recall do blocking);
  correto     = o escolhido é o CNS certo;
  recall      = corretos / consultas;  precisão = corretos / consultas vinculadas.

Duas condições de pool: a UF inteira (milhões; colisão máxima) e um pool do
tamanho de uma coorte de doença (1% dos pacientes da UF, sorteados com sal; o
pool real da cascata no estudo de caso é a coorte, milhares). R2 (nasc+CEP) não é
testável: nenhum cache nacional do BI tem CEP.

Uso:
    python scripts/cascata_registro_retido_uf.py --extrair --ufs 28 50 12
    python scripts/cascata_registro_retido_uf.py --rodar --ufs 28 50 12

Saídas em dados/stats/cascata_registro_retido/ (+ outputs/):
    cascata_registro_retido_ate2025.csv
"""
from __future__ import annotations

import argparse
import shutil

import duckdb
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths
from aux_cns import registrar_macros

CASC = _paths.CACHE / "cascade_bi_estabilidade_ate2025"
EXTR = _paths.CACHE / "cascata_registro_retido"
OUT = _paths.STATS / "cascata_registro_retido"
AGREGADOS = _paths.STATS.parents[1] / "outputs"
SAL = 20261008
REGRAS = {"R1": ["dtnasc", "cnes"], "R3": ["dtnasc", "sexo", "mun_res"], "R4": ["dtnasc", "sexo", "mun_ubs"]}
ORDENS = {"produção (R1→R3→R4)": ["R1", "R3", "R4"], "C50 (R3→R1→R4)": ["R3", "R1", "R4"]}


def conectar(tag: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=3")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{_paths.CACHE / 'tmp_duck' / ('cascata_' + tag)}'")
    registrar_macros(con)
    return con


def extrair(ufs: list[str]) -> None:
    EXTR.mkdir(parents=True, exist_ok=True)
    lista = ", ".join(f"'{u}'" for u in ufs)
    for k in range(24):
        destino = EXTR / f"part={k}.parquet"
        if destino.exists():
            continue
        con = conectar(f"x{k}")
        con.execute(f"""COPY (
            SELECT cns, dtnasc, sexo, mun_res, mun_ubs, cnes, substr(mun_res, 1, 2) AS uf,
                   {k}::BIGINT * 10000000000 + ROW_NUMBER() OVER () AS rid
            FROM read_parquet('{(CASC / f"part={k}" / "*.parquet").as_posix()}')
            WHERE cns_status = 'valido_candidato' AND cns_dv_valido(trim(cns))
              AND substr(mun_res, 1, 2) IN ({lista})
        ) TO '{destino}.tmp' (FORMAT parquet)""")
        (EXTR / f"part={k}.parquet.tmp").rename(destino)
        n = con.execute(f"SELECT COUNT(*) FROM '{destino}'").fetchone()[0]
        print(f"  extração: partição {k} ({n:,} registros)", flush=True)


def _chave(cols: list[str], a: str = "") -> str:
    p = f"{a}." if a else ""
    nn = " AND ".join(f"{p}{c} IS NOT NULL AND trim({p}{c}) <> ''" for c in cols)
    return f"CASE WHEN {nn} THEN {' || chr(31) || '.join(p + c for c in cols)} END"


def rodar_uf(uf: str, pool_frac: float | None, nasc_min: int | None = None) -> list[dict]:
    con = conectar(f"r{uf}")
    con.execute(f"CREATE TEMP TABLE r AS SELECT * FROM '{EXTR}/part=*.parquet' WHERE uf = '{uf}'")
    if nasc_min is not None:  # pool pediátrico: só nascidos a partir de `nasc_min`, como numa coorte infantil
        con.execute(f"DELETE FROM r WHERE TRY_CAST(substr(dtnasc, 1, 4) AS INT) IS NULL OR TRY_CAST(substr(dtnasc, 1, 4) AS INT) < {nasc_min}")
    if pool_frac is not None:  # pool do tamanho de uma coorte: só uma fração dos pacientes
        con.execute(f"""DELETE FROM r WHERE (abs(hash(cns || '{SAL}')) % 10000) >= {int(pool_frac * 10000)}""")
    # consulta = 1 registro retido de cada paciente com ≥ 2 registros
    # GROUP BY em vez de janela: janela sobre dezenas de milhões de linhas não
    # derrama para disco (estourou 3 GB em SE); o registro retido é o de menor
    # (hash com sal, rid) — ordem total, determinística.
    con.execute(f"""CREATE TEMP TABLE q0 AS
        SELECT cns, arg_min(rid, (hash(rid, {SAL}), rid)) AS rid FROM r GROUP BY cns HAVING COUNT(*) >= 2""")
    con.execute("CREATE TEMP TABLE q AS SELECT r.* FROM r JOIN q0 USING (cns, rid)")
    assert con.execute("SELECT COUNT(*) FROM q").fetchone()[0] == con.execute("SELECT COUNT(*) FROM q0").fetchone()[0]
    con.execute("CREATE TEMP TABLE pool AS SELECT r.* FROM r ANTI JOIN q USING (rid)")
    n_q = con.execute("SELECT COUNT(*) FROM q").fetchone()[0]
    n_pac_pool = con.execute("SELECT COUNT(DISTINCT cns) FROM pool").fetchone()[0]
    for nome, cols in REGRAS.items():
        con.execute(f"""CREATE TEMP TABLE lk_{nome} AS
            SELECT {_chave(cols)} AS k, MIN(cns) AS escolhido, COUNT(DISTINCT cns) AS n_cand
            FROM pool WHERE {_chave(cols)} IS NOT NULL GROUP BY 1""")
        con.execute(f"""CREATE TEMP TABLE has_{nome} AS
            SELECT DISTINCT {_chave(cols)} AS k, cns FROM pool WHERE {_chave(cols)} IS NOT NULL""")
        con.execute(f"""CREATE TEMP TABLE res_{nome} AS
            SELECT q.rid, q.cns AS verdadeiro, l.escolhido, l.n_cand, (h.cns IS NOT NULL) AS encontrado
            FROM q LEFT JOIN lk_{nome} l ON l.k = {_chave(cols, 'q')}
                   LEFT JOIN has_{nome} h ON h.k = {_chave(cols, 'q')} AND h.cns = q.cns""")
    linhas = []
    pool_rot = "UF inteira" if pool_frac is None else f"{pool_frac:.0%} dos pacientes (tamanho de coorte)"
    if nasc_min is not None:
        pool_rot += f", nascidos >= {nasc_min} (pediátrico)"
    base = {"uf": uf, "pool": pool_rot, "consultas": n_q, "pacientes_no_pool": n_pac_pool}
    for nome in REGRAS:
        v, enc, cor, amb = con.execute(f"""SELECT COUNT(escolhido), COALESCE(SUM(encontrado::INT), 0),
                COALESCE(SUM((escolhido = verdadeiro)::INT), 0), COALESCE(SUM((n_cand > 1)::INT), 0)
                FROM res_{nome}""").fetchone()
        linhas.append(base | {"ordem": "sozinha", "regra": nome, "vinculadas": v, "encontrado": enc,
                              "corretas": cor, "ambiguas": amb})
    for ordem, regs in ORDENS.items():
        coalesce = lambda col: "COALESCE(" + ", ".join(f"{r}.{col}" for r in regs) + ")"  # noqa: E731
        decisor = "CASE " + " ".join(f"WHEN {r}.escolhido IS NOT NULL THEN '{r}'" for r in regs) + " END"
        joins = " ".join(f"JOIN res_{r} {r} USING (rid)" for r in regs[1:])
        df = con.execute(f"""
            SELECT {decisor} AS regra, COUNT(*) AS consultas_na_regra,
                   COALESCE(SUM(({coalesce('escolhido')} = {regs[0]}.verdadeiro)::INT), 0) AS corretas,
                   COALESCE(SUM(({coalesce('n_cand')} > 1)::INT), 0) AS ambiguas
            FROM res_{regs[0]} {regs[0]} {joins} GROUP BY 1""").df()
        for _, x in df.iterrows():
            linhas.append(base | {"ordem": ordem, "regra": x.regra or "sem vínculo",
                                  "vinculadas": 0 if x.regra is None else int(x.consultas_na_regra),
                                  "encontrado": None, "corretas": int(x.corretas),
                                  "ambiguas": int(x.ambiguas)})
    print(f"  UF {uf}, pool {pool_rot}: {n_q:,} consultas, {n_pac_pool:,} pacientes no pool", flush=True)
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufs", nargs="+", required=True)
    ap.add_argument("--extrair", action="store_true")
    ap.add_argument("--rodar", action="store_true")
    ap.add_argument("--so-agregar", action="store_true",
                    help="não roda nada: reconstrói os agregados a partir das linhas brutas do CSV")
    ap.add_argument("--nasc-min", type=int, default=None,
                    help="só o pool pediátrico (nascidos >= ano), acrescentado ao CSV existente")
    a = ap.parse_args()
    if a.extrair:
        extrair(a.ufs)
    if not a.rodar and not a.so_agregar:
        return
    linhas = []
    if not a.so_agregar:
        for uf in a.ufs:
            if a.nasc_min is not None:
                linhas += rodar_uf(uf, 0.01, a.nasc_min)
            else:
                for frac in (None, 0.01):
                    linhas += rodar_uf(uf, frac)
    t = pd.DataFrame(linhas)
    if (a.nasc_min is not None or a.so_agregar) and (OUT / "cascata_registro_retido_ate2025.csv").exists():
        # só as linhas BRUTAS (por UF, por regra); os agregados são recomputados abaixo
        antigo = pd.read_csv(OUT / "cascata_registro_retido_ate2025.csv")
        antigo = antigo[(antigo.uf != "todas") & (antigo.regra != "cascata (acumulada)")]
        if a.nasc_min is not None:
            antigo = antigo[~antigo.pool.str.contains("pediátrico")]
        antigo = antigo.drop(columns=[c for c in ("recall", "precisao") if c in antigo])
        t = pd.concat([antigo, t], ignore_index=True).drop_duplicates(["uf", "pool", "ordem", "regra"])
    t["recall"] = t.corretas / t.consultas
    t["precisao"] = t.corretas / t.vinculadas.where(t.vinculadas > 0)
    # acumulado da cascata por UF/pool/ordem
    cas = (t[t.ordem != "sozinha"].groupby(["uf", "pool", "ordem", "consultas"], as_index=False)
           [["vinculadas", "corretas", "ambiguas"]].sum())
    cas["regra"] = "cascata (acumulada)"
    cas["recall"] = cas.corretas / cas.consultas
    cas["precisao"] = cas.corretas / cas.vinculadas
    t = pd.concat([t, cas], ignore_index=True)
    # agregado entre UFs
    ag = (t.groupby(["pool", "ordem", "regra"], as_index=False)[["consultas", "vinculadas", "corretas", "ambiguas"]]
          .sum().assign(uf="todas"))
    ag["recall"] = ag.corretas / ag.consultas
    ag["precisao"] = ag.corretas / ag.vinculadas.where(ag.vinculadas > 0)
    t = pd.concat([t, ag], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "cascata_registro_retido_ate2025.csv", index=False)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "cascata_registro_retido_ate2025.csv", AGREGADOS / "cascata_registro_retido_ate2025.csv")
    print(ag.round(4).to_string(index=False))
    print(f"✅ {OUT / 'cascata_registro_retido_ate2025.csv'}")


if __name__ == "__main__":
    main()
