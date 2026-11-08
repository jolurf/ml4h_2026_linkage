"""Diagnóstico dos parâmetros u e m do Fellegi–Sunter.

Três correções sobre o estimador ingênuo dos parâmetros, todas medidas, nenhuma assumida:

1. **u sem auto-par.** `4_probabilistic_joining` estima u por Σn²/N² sobre o
   perfil (uma linha por paciente). Isso amostra pares COM reposição: cada
   paciente entra pareado consigo mesmo, termo 1/N. u é P(concordar | pessoas
   DIFERENTES), então o estimador certo é Σn(n−1)/(N(N−1)). No pool nacional
   (N=2,3×10⁸) a diferença é pequena; na coorte (N=6.605) o 1/N domina o u
   conjunto.
2. **m conjunto medido.** `4_estimar_m_goldstandard.m_composto` multiplica os m
   por campo. Aqui o m da chave composta é medido como o de um campo: a chave
   concatenada, a mesma taxa de concordância por CNS ponderada por paciente.
3. **Mesma criança sob dois CNS** (só coorte). Pares de CNS distintos que
   concordam em nascimento+sexo+município entram no u como "pessoas diferentes".
   Faixas de evidência de que são a mesma criança: mesmo CNES; mesmo CNES no mesmo
   ano; mesmo CNES na mesma competência (mês). A faixa estrita (mês) é a usada
   para recomputar u sem duplicatas; as outras são publicadas ao lado.

Uso:
    python scripts/fs_diagnostico_u_m.py --escopo coorte
    python scripts/fs_diagnostico_u_m.py --escopo nacional   # ~1 h, DuckDB em disco
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import duckdb
import pandas as pd

import _raiz  # noqa: F401  (ancora na raiz e põe src/ no path)

RAIZ = Path(__file__).resolve().parents[1]
AGREGADOS_OUT = RAIZ / "outputs"
MAX_DT = 3  # o mesmo filtro anti-placeholder do 4_estimar_m_goldstandard

#: chaves da Tabela 5, expressas sobre o perfil (u) e sobre o cache do BI (m)
CHAVES = {
    "data_nascimento": (["data_nascimento"], ["dtnasc"]),
    "sexo": (["sexo"], ["sexo"]),
    "mun_res": (["mun_res_principal"], ["mun_res"]),
    "nasc+sexo": (["data_nascimento", "sexo"], ["dtnasc", "sexo"]),
    "nasc+mun_res": (["data_nascimento", "mun_res_principal"], ["dtnasc", "mun_res"]),
    "nasc+sexo+mun_res": (["data_nascimento", "sexo", "mun_res_principal"],
                          ["dtnasc", "sexo", "mun_res"]),
}
SIMPLES = ("data_nascimento", "sexo", "mun_res")


def _expr(cols: list[str], alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return " || '_' || ".join(f"{p}{c}" for c in cols)


def escopo_paths(escopo: str) -> dict:
    if escopo == "coorte":
        base = RAIZ / "projetos" / "Nefroblastoma_sensibilidade"
        return {"perfil": f"'{base}/stats/perfil_paciente/*.parquet'",
                "cache": base / "stats" / "cache_bi", "npart": 1,
                "out": base / "experimento", "proj": base}
    return {"perfil": f"'{RAIZ}/dados/stats/perfil_paciente/*.parquet'",
            "cache": RAIZ / "dados" / "cache" / "base_bi", "npart": 24,
            "out": RAIZ / "dados" / "stats" / "fs_diagnostico", "proj": None}


def conectar() -> duckdb.DuckDBPyConnection:
    tmp = RAIZ / ".duckdb_tmp_fs"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def u_tabela(con, perfil: str) -> pd.DataFrame:
    """u por chave, com e sem auto-par. `perfil` é uma expressão FROM."""
    linhas = []
    for nome, (cols, _) in CHAVES.items():
        e = _expr(cols)
        tot, ss, sn1 = con.execute(f"""
            WITH c AS (SELECT {e} AS k, COUNT(*)::DOUBLE AS n FROM {perfil}
                       WHERE ({e}) IS NOT NULL GROUP BY k)
            SELECT SUM(n), SUM(n*n), SUM(n*(n-1)) FROM c""").fetchone()
        linhas.append({"chave": nome, "N_pool": int(tot),
                       "u_com_autopar": ss / (tot * tot),
                       "u_sem_autopar": sn1 / (tot * (tot - 1)),
                       "pares_concordantes": sn1 / 2})
    u = pd.DataFrame(linhas).set_index("chave")
    for col in ("u_com_autopar", "u_sem_autopar"):
        u[f"produto_{col}"] = None
        for nome, (cols, _) in CHAVES.items():
            if len(cols) > 1:
                partes = [SIMPLES[i] for i, c in enumerate(["data_nascimento", "sexo",
                          "mun_res_principal"]) if c in cols]
                u.loc[nome, f"produto_{col}"] = math.prod(u.loc[p, col] for p in partes)
    u["razao_com_autopar"] = u["u_com_autopar"] / u["produto_u_com_autopar"].astype(float)
    u["razao_sem_autopar"] = u["u_sem_autopar"] / u["produto_u_sem_autopar"].astype(float)
    return u


def m_tabela(con, cache: Path, npart: int) -> pd.DataFrame:
    """m por chave (simples e composta) pela taxa de concordância por CNS,
    ponderada por paciente — a mesma conta do `m_field` do notebook."""
    acc = {nome: [0.0, 0] for nome in CHAVES}
    for k in range(npart):
        bp = f"read_parquet('{(cache / f'part={k}' / '*.parquet').as_posix()}')"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE b AS
            SELECT x.* FROM {bp} x SEMI JOIN (SELECT cns FROM {bp} GROUP BY cns
                HAVING COUNT(DISTINCT dtnasc) <= {MAX_DT}) ok USING (cns)""")
        for nome, (_, cols) in CHAVES.items():
            e = _expr(cols)
            nn = " AND ".join(f"{c} IS NOT NULL AND trim({c}) <> ''" for c in cols)
            s, n = con.execute(f"""
                WITH v AS (SELECT cns, {e} AS val, COUNT(*)::DOUBLE AS c FROM b
                           WHERE {nn} GROUP BY cns, val),
                     p AS (SELECT cns, SUM(c) AS n, SUM(c*c) AS ss FROM v GROUP BY cns)
                SELECT COALESCE(SUM((ss-n)/(n*(n-1))),0), COUNT(*) FROM p WHERE n > 1
            """).fetchone()
            acc[nome][0] += s
            acc[nome][1] += n
        print(f"  m: partição {k + 1}/{npart}", flush=True)
    m = pd.DataFrame([{"chave": k, "m_medido": s / n, "n_pacientes": n}
                      for k, (s, n) in acc.items()]).set_index("chave")
    m["m_produto"] = None
    for nome, (_, cols) in CHAVES.items():
        if len(cols) > 1:
            m.loc[nome, "m_produto"] = math.prod(
                m.loc[SIMPLES[["dtnasc", "sexo", "mun_res"].index(c)], "m_medido"] for c in cols)
    return m


def pares_duplicados(con, paths: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pares de CNS distintos com nascimento+sexo+município iguais, e as faixas de
    evidência de que são a mesma criança."""
    proj = paths["proj"]
    con.execute(f"""CREATE OR REPLACE TEMP TABLE p AS
        SELECT *, split_part(id_paciente, '_', 1) AS cns FROM {paths['perfil']}""")
    con.execute("""CREATE OR REPLACE TEMP TABLE pares AS
        SELECT a.id_paciente AS id_a, b.id_paciente AS id_b, a.cns AS cns_a, b.cns AS cns_b,
               a.data_nascimento AS dtnasc, a.sexo, a.mun_res_principal AS mun
        FROM p a JOIN p b ON a.data_nascimento = b.data_nascimento AND a.sexo = b.sexo
             AND a.mun_res_principal = b.mun_res_principal AND a.id_paciente < b.id_paciente""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE bi AS
        SELECT DISTINCT cns, cnes, ano FROM read_parquet('{paths['cache']}/part=0/*.parquet')
        WHERE cns IN (SELECT cns_a FROM pares UNION SELECT cns_b FROM pares)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rhc AS
        SELECT DISTINCT id_paciente, RHC_ID FROM '{proj}/join_rhc_sus.parquet'""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE sia AS
        SELECT DISTINCT cns, cnes_unidade AS cnes, substr(CAST(data_atendimento AS VARCHAR),1,7) AS comp
        FROM '{proj}/df_final_scored.parquet'
        WHERE cns IN (SELECT cns_a FROM pares UNION SELECT cns_b FROM pares) AND cnes_unidade IS NOT NULL""")

    def faixa(tab: str, cond: str) -> str:
        return (f"(SELECT DISTINCT cns_a, cns_b FROM pares JOIN {tab} x ON x.cns = cns_a "
                f"JOIN {tab} y ON y.cns = cns_b AND {cond})")
    det = con.execute(f"""
        SELECT pares.*,
               f1.cns_a IS NOT NULL AS mesmo_cnes,
               f2.cns_a IS NOT NULL AS mesmo_cnes_ano,
               f3.cns_a IS NOT NULL AS mesmo_cnes_mes,
               f0.cns_a IS NOT NULL AS mesmo_rhc_id
        FROM pares
        LEFT JOIN {faixa('bi', 'x.cnes = y.cnes')} f1 USING (cns_a, cns_b)
        LEFT JOIN {faixa('bi', 'x.cnes = y.cnes AND x.ano = y.ano')} f2 USING (cns_a, cns_b)
        LEFT JOIN {faixa('sia', 'x.cnes = y.cnes AND x.comp = y.comp')} f3 USING (cns_a, cns_b)
        LEFT JOIN (SELECT DISTINCT pares.cns_a, pares.cns_b FROM pares
                   JOIN rhc x ON x.id_paciente = pares.id_a JOIN rhc y ON y.id_paciente = pares.id_b
                   AND x.RHC_ID = y.RHC_ID) f0 USING (cns_a, cns_b)
        ORDER BY dtnasc, mun, cns_a, cns_b""").df()
    n = len(det)
    res = pd.DataFrame([
        {"criterio": "CNS distintos, mesmo nascimento+sexo+município", "n_pares": n, "pct": 100.0},
        *[{"criterio": rot, "n_pares": int(det[c].sum()), "pct": 100 * det[c].mean()}
          for c, rot in [("mesmo_rhc_id", "mesmo registro de tumor no RHC (duplicata definitiva)"),
                         ("mesmo_cnes", "mesmo CNES em algum momento"),
                         ("mesmo_cnes_ano", "mesmo CNES no mesmo ano"),
                         ("mesmo_cnes_mes", "mesmo CNES na mesma competência (mês)")]]])
    return det, res


def u_sem_duplicatas(con, det: pd.DataFrame, faixa: str) -> pd.DataFrame:
    """Remove do pool o id maior de cada par adjudicado e recalcula u."""
    remover = set(det.loc[det[faixa], "id_b"])
    con.register("_remover", pd.DataFrame({"id_paciente": sorted(remover)}))
    con.execute("CREATE OR REPLACE TEMP TABLE p_dedup AS SELECT * FROM p ANTI JOIN _remover USING (id_paciente)")
    return u_tabela(con, "p_dedup")


def gravar(df: pd.DataFrame, destino: Path, nome: str) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    f = destino / nome
    df.to_csv(f, index=True)
    AGREGADOS_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, AGREGADOS_OUT / nome)
    print(f"✅ {f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--escopo", choices=("coorte", "nacional"), required=True)
    ap.add_argument("--sem-m", action="store_true", help="pula o m (a parte cara no nacional)")
    a = ap.parse_args()
    P = escopo_paths(a.escopo)
    con = conectar()
    sufixo = a.escopo

    u = u_tabela(con, P["perfil"])
    # o estimador antigo tem de reproduzir o que foi publicado/gravado
    ref = {"coorte": 1.7588125436594326e-4, "nacional": 1.362897131129071e-7}[a.escopo]
    assert abs(u.loc["nasc+sexo+mun_res", "u_com_autopar"] / ref - 1) < 1e-9, "u antigo não reproduz"
    print(u[["N_pool", "u_com_autopar", "u_sem_autopar", "razao_com_autopar", "razao_sem_autopar"]])
    gravar(u, P["out"], f"fs_u_corrigido_{sufixo}.csv")

    if a.escopo == "coorte":
        det, res = pares_duplicados(con, P)
        assert len(det) == round(u.loc["nasc+sexo+mun_res", "pares_concordantes"]), \
            "nº de pares concordantes não bate com Σn(n−1)/2"
        print(res)
        gravar(det.set_index("id_a"), P["out"], "fs_pares_duplicados_coorte.csv")
        gravar(res.set_index("criterio"), P["out"], "fs_duplicatas_resumo_coorte.csv")
        linhas = []
        for faixa in ("mesmo_rhc_id", "mesmo_cnes_mes", "mesmo_cnes_ano", "mesmo_cnes"):
            ud = u_sem_duplicatas(con, det, faixa)
            linhas.append({"duplicatas_removidas_por": faixa,
                           "N_pool": ud.loc["nasc+sexo+mun_res", "N_pool"],
                           "u_conjunto": ud.loc["nasc+sexo+mun_res", "u_sem_autopar"],
                           "u_produto": ud.loc["nasc+sexo+mun_res", "produto_u_sem_autopar"],
                           "razao": ud.loc["nasc+sexo+mun_res", "razao_sem_autopar"]})
        dd = pd.DataFrame(linhas).set_index("duplicatas_removidas_por")
        print(dd)
        gravar(dd, P["out"], "fs_u_sem_duplicatas_coorte.csv")

    if not a.sem_m:
        m = m_tabela(con, P["cache"], P["npart"])
        m = m.join(u[["u_com_autopar", "u_sem_autopar"]])
        m["peso_bits_publicado"] = m.apply(
            lambda r: math.log2((r.m_produto or r.m_medido) / r.u_com_autopar), axis=1)
        m["peso_bits_corrigido"] = m.apply(lambda r: math.log2(r.m_medido / r.u_sem_autopar), axis=1)
        print(m)
        gravar(m, P["out"], f"fs_m_conjunto_{sufixo}.csv")


if __name__ == "__main__":
    main()
