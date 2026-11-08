#!/usr/bin/env python3
"""Etapa 2 do protocolo do estudo: Tabela 1 nacional (SIA/BI) — população
com CNS confiável vs sem, nas dimensões demográficas/geográficas/assistenciais,
com teste qui-quadrado formalizando que a ausência do CNS não é aleatória
(MNAR / MAR estruturado).

**"CNS confiável" aqui é a definição composta, não só formato.** Reusa TUDO
que já foi estabelecido nesta análise, cada critério motivado por um achado
concreto, não por suposição:
  1. presente (`CNS_PAC` não nulo/vazio) — `analisar_cns_faltante_bi.py`
  2. bem formado (15 caracteres no alfabeto cifrado) — idem
  3. não é placeholder de dígito repetido (`chr(123)*15` etc.) — idem
  4. não associado a mais de `MAX_DT` datas de nascimento distintas — idem
     (CNS genérico/institucional reusado; Etapa 1 achou 4 exemplos assim,
     82K-652K reusos cada, e TODOS tinham dígito verificador válido — critério
     4 pega o que o critério 5 sozinho NÃO pegaria)
  5. passa no dígito verificador oficial do CNS — `validar_cns_bi.py`
     (Etapa 1: 92,28% dos CNS bem formados passam; o resto tem aparência de
     CNS real mas não é um número válido)

**SIH e SIM não entram na comparação "preenchimento por base".** É fato
estrutural: só o SIA/BI tem CNS na mesma linha que
nascimento+CID. Nas outras duas a "taxa de preenchimento" seria 0% por
desenho, não por qualidade de dado — reportar isso lado a lado do SIA
mascararia a causa. A Tabela 1 deste script é só do SIA/BI.

**Dimensões cobertas** (Etapa 2 do protocolo): raça/cor, faixa etária, sexo
(demográficas); macro-região, UF (geográficas — porte de município fica de
fora: teria que casar `MUNPAC` com `tabelas/IBGE.parquet` e o
formato dos códigos dos dois não foi conferido — arriscar um join errado é
pior que não ter a dimensão, então fica marcado como pendência, não
inventado); tipo de estabelecimento (`TPUPS`), complexidade (`COMPLEX`) e
ano do atendimento (assistenciais).

**Reusa a infraestrutura já construída, não reinventa.** As funções de
região/UF/raça/faixa etária vêm de `analisar_cns_faltante_bi` (mesmo
de-para, mesmo rótulo). A lista de CNS com dígito verificador válido vem de
`validar_cns_bi` (`dados/cache/faltantes_cns_bi/cns_validos/`) — reaproveita
o particionamento por `hash(CNS) % NPART` já materializado lá, então o join
por partição é pequeno (mesmo truque que resolveu o OOM da Etapa B).

Duas fases, mesmo padrão de isolamento por subprocesso já validado nas
etapas anteriores (cada partição roda sozinha, resumível):
  1. cache — 1 varredura do BI bruto, com as colunas extras (SEXOPAC, TPUPS,
     COMPLEX), particionado por hash(CNS) igual ao `cns_validos`.
  2. por partição — resolve confiabilidade composta e agrega por dimensão.
  3. teste qui-quadrado por dimensão (via scipy, roda na tabela agregada —
     pequena, sem risco de memória).

Uso:
    python scripts/tabela1_cns_nacional.py
    python scripts/tabela1_cns_nacional.py --max-dt 3 --npart 24 --memory-limit 6GB

Saídas em `dados/stats/tabela1_cns_bi/`:
    tabela1_nacional.csv       long-format: dimensao, categoria, com_cns, sem_cns, pct_sem_cns
    testes_qui_quadrado.csv    dimensao, chi2, dof, p_valor, cramers_v, n
    relatorio_tabela1.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import _raiz  # noqa: F401
import paths as _paths
from analisar_cns_faltante_bi import (BI_GLOB, CNS_ALFABETO_VALIDO, CNS_WIDTH,
                                       RACACOR_MAP, _faixa_etaria_case,
                                       _idade_anos_case, _regiao_uf_case)

CACHE_DIR = _paths.CACHE / "tabela1_cns_bi"
OUT_DIR = _paths.STATS / "tabela1_cns_bi"

DIMENSOES = ["sexo", "raca_cor", "faixa_etaria", "regiao", "uf", "tpups", "complex", "ano"]


def _sufixo(ano_max: int | None) -> str:
    return "" if ano_max is None else f"_ate{ano_max}"


def _validos_dir(ano_max: int | None) -> Path:
    # aponta pra saída (já com sufixo) da Etapa 1 (`validar_cns_bi.py`) —
    # a lista de CNS válidos tem que vir da MESMA janela de ano, senão um
    # CNS "válido" que só apareceu em 2026 entraria como confiável aqui
    # mesmo os registros de 2026 estando fora do filtro desta tabela.
    return _paths.CACHE / "faltantes_cns_bi" / f"cns_validos{_sufixo(ano_max)}"


def _parcial_dir(ano_max: int | None) -> Path:
    return OUT_DIR / f"_parcial{_sufixo(ano_max)}"


def _materializar_cache(memory_limit: str, threads: int, npart: int) -> None:
    if CACHE_DIR.exists():
        print(f"  ↺ reusando cache existente em {CACHE_DIR}/ (apague a pasta p/ refazer)")
        return
    print(f"  materializando cache ({npart} partições) — 1 varredura do BI bruto...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET temp_directory='./.duckdb_tmp_tabela1_cache'")
    con.execute("SET preserve_insertion_order=false")

    bem_formado = f"regexp_matches(trim(CNS_PAC), '{CNS_ALFABETO_VALIDO}')"
    repetido = f"repeat(substr(trim(CNS_PAC), 1, 1), {CNS_WIDTH}) = trim(CNS_PAC)"

    t0 = time.time()
    con.execute(f"""
      COPY (
        SELECT
          trim(CNS_PAC) AS cns,
          DTNASC AS dtnasc,
          RACACOR AS racacor,
          TPIDADEPAC AS tp_idade,
          IDADEPAC AS idade_cod,
          MUNPAC AS mun_res,
          SEXOPAC AS sexo,
          TPUPS AS tpups,
          COMPLEX AS complex,
          ANO AS ano,
          CASE
            WHEN CNS_PAC IS NULL OR length(trim(CNS_PAC)) = 0 THEN 'ausente'
            WHEN NOT {bem_formado} THEN 'malformado'
            WHEN {repetido} THEN 'placeholder_repetido'
            ELSE 'valido_candidato'
          END AS cns_status,
          -- mesmo esquema de partição do cns_validos da Etapa 1: precisa
          -- alinhar pra o semi-join por partição funcionar sem embaralhar de novo.
          CASE
            WHEN CNS_PAC IS NOT NULL AND {bem_formado} AND NOT ({repetido})
              THEN abs(hash(trim(CNS_PAC))) % {npart}
            ELSE CAST(floor(random() * {npart}) AS INTEGER)
          END AS part
        FROM read_parquet('{BI_GLOB}', union_by_name=true, hive_partitioning=true)
      ) TO '{CACHE_DIR.as_posix()}'
        (FORMAT parquet, PARTITION_BY (part), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
    """)
    print(f"  ✅ cache em {CACHE_DIR}/ ({time.time()-t0:.0f}s)", flush=True)


def _processar_particao(k: int, memory_limit: str, threads: int, max_dt: int, ano_max: int | None) -> None:
    pasta = CACHE_DIR / f"part={k}"
    if not pasta.exists():
        print(f"  partição {k}: pasta não existe, pulando")
        return

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    filtro_ano = f"AND ano <= {ano_max}" if ano_max is not None else ""
    bp = f"read_parquet('{(pasta / '*.parquet').as_posix()}')"

    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE _ruim AS
      SELECT cns FROM {bp}
      WHERE cns_status = 'valido_candidato' {filtro_ano}
      GROUP BY cns
      HAVING COUNT(DISTINCT dtnasc) > {max_dt}
    """)

    validos_path = _validos_dir(ano_max) / f"part={k}.parquet"
    tem_validos = validos_path.exists()
    if tem_validos:
        con.execute(f"""
          CREATE OR REPLACE TEMP TABLE _validos AS
          SELECT cns FROM read_parquet('{validos_path.as_posix()}')
        """)
    digito_valido_expr = ("cns IN (SELECT cns FROM _validos)" if tem_validos else "FALSE")

    regiao_case = _regiao_uf_case(_paths.JSONS / "region_state.json", "substr(mun_res, 1, 2)", "regiao")
    uf_case = _regiao_uf_case(_paths.JSONS / "num_state.json", "substr(mun_res, 1, 2)", "uf")
    idade_expr = _idade_anos_case("tp_idade", "idade_cod")
    faixa_case = _faixa_etaria_case(idade_expr)
    raca_linhas = "\n    ".join(f"WHEN racacor = '{cod}' THEN '{lbl}'" for cod, lbl in RACACOR_MAP.items())
    raca_case = f"CASE\n    {raca_linhas}\n    ELSE 'Não informado' END"

    df = con.execute(f"""
      SELECT
        sexo,
        {raca_case} AS raca_cor,
        {faixa_case} AS faixa_etaria,
        {regiao_case},
        {uf_case},
        tpups,
        complex,
        ano,
        (cns_status = 'valido_candidato'
         AND cns NOT IN (SELECT cns FROM _ruim)
         AND {digito_valido_expr}) AS tem_cns_confiavel,
        COUNT(*) AS registros
      FROM {bp}
      WHERE 1=1 {filtro_ano}
      GROUP BY ALL
    """).df()

    df.to_parquet(_parcial_dir(ano_max) / f"part={k}.parquet", index=False)
    print(f"  partição {k}: {len(df):,} linhas de resumo "
          f"({'com' if tem_validos else 'SEM'} lista de CNS válidos)", flush=True)


def _agregar_dimensao(agregado: pd.DataFrame, coluna: str) -> pd.DataFrame:
    t = (agregado.groupby([coluna, "tem_cns_confiavel"], as_index=False)["registros"].sum())
    piv = t.pivot(index=coluna, columns="tem_cns_confiavel", values="registros").fillna(0)
    piv.columns = ["sem_cns" if not c else "com_cns" for c in piv.columns]
    for col in ("com_cns", "sem_cns"):
        if col not in piv.columns:
            piv[col] = 0
    piv["total"] = piv["com_cns"] + piv["sem_cns"]
    piv["pct_sem_cns"] = 100 * piv["sem_cns"] / piv["total"]
    piv = piv.reset_index().rename(columns={coluna: "categoria"})
    piv.insert(0, "dimensao", coluna)
    return piv.sort_values("pct_sem_cns", ascending=False)


def _teste_qui_quadrado(agregado: pd.DataFrame, coluna: str) -> dict:
    t = (agregado.groupby([coluna, "tem_cns_confiavel"])["registros"].sum().unstack(fill_value=0))
    if t.shape[0] < 2 or t.shape[1] < 2:
        return {"dimensao": coluna, "chi2": np.nan, "dof": np.nan, "p_valor": np.nan,
                "cramers_v": np.nan, "n": int(t.values.sum())}
    chi2, p, dof, _ = chi2_contingency(t.values)
    n = t.values.sum()
    k_menor = min(t.shape) - 1
    cramers_v = np.sqrt((chi2 / n) / k_menor) if k_menor > 0 else np.nan
    return {"dimensao": coluna, "chi2": chi2, "dof": dof, "p_valor": p,
            "cramers_v": cramers_v, "n": int(n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npart", type=int, default=24)
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-dt", type=int, default=3)
    ap.add_argument("--ano-max", type=int, default=None,
                     help="restringe a janela a ano<=X (ex.: 2025); exige que a Etapa 1 já tenha "
                          "rodado com o MESMO --ano-max")
    ap.add_argument("--processar-particao", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parcial_dir = _parcial_dir(args.ano_max)
    parcial_dir.mkdir(parents=True, exist_ok=True)

    if args.processar_particao is not None:
        _processar_particao(args.processar_particao, args.memory_limit, args.threads, args.max_dt, args.ano_max)
        return

    validos_dir = _validos_dir(args.ano_max)
    if not validos_dir.exists() or not any(validos_dir.glob("part=*.parquet")):
        raise SystemExit(f"ERRO: {validos_dir} não existe — rode primeiro "
                          f"`python scripts/validar_cns_bi.py --so-digito"
                          f"{' --ano-max ' + str(args.ano_max) if args.ano_max else ''}` (Etapa 1).")

    print("Fase 1 — cache nacional (BI + colunas extras: sexo, tipo de estabelecimento, complexidade)")
    _materializar_cache(args.memory_limit, args.threads, args.npart)

    print("\nFase 2 — resolução de confiabilidade + agregação, por partição")
    for k in range(args.npart):
        if (parcial_dir / f"part={k}.parquet").exists():
            print(f"  partição {k+1}/{args.npart} já concluída (resumindo) — pulando")
            continue
        t0 = time.time()
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--processar-particao", str(k),
               "--memory-limit", args.memory_limit, "--threads", str(args.threads),
               "--max-dt", str(args.max_dt)]
        if args.ano_max is not None:
            cmd += ["--ano-max", str(args.ano_max)]
        subprocess.run(cmd, check=True)
        print(f"  partição {k+1}/{args.npart} ok ({time.time()-t0:.0f}s)", flush=True)

    print("\nFase 3 — Tabela 1 + testes qui-quadrado")
    partes = [pd.read_parquet(f) for f in sorted(parcial_dir.glob("part=*.parquet"))]
    agregado = pd.concat(partes, ignore_index=True)

    tabela1 = pd.concat([_agregar_dimensao(agregado, c) for c in DIMENSOES], ignore_index=True)
    tabela1.to_csv(OUT_DIR / f"tabela1_nacional{_sufixo(args.ano_max)}.csv", index=False)

    testes = pd.DataFrame([_teste_qui_quadrado(agregado, c) for c in DIMENSOES])
    testes.to_csv(OUT_DIR / f"testes_qui_quadrado{_sufixo(args.ano_max)}.csv", index=False)

    total_com = agregado.loc[agregado["tem_cns_confiavel"], "registros"].sum()
    total_geral = agregado["registros"].sum()
    print(f"\nCNS confiável (definição composta): {total_com:,} / {total_geral:,} "
          f"({100*total_com/total_geral:.2f}%)")

    titulo_janela = f" (até {args.ano_max})" if args.ano_max else ""
    linhas = [f"# Tabela 1 nacional — CNS confiável vs não (SIA/BI){titulo_janela}", "",
              f"**{total_geral:,} registros**; **{total_com:,} ({100*total_com/total_geral:.2f}%)** "
              f"com CNS confiável pela definição composta (presente, bem formado, não-placeholder, "
              f"≤{args.max_dt} datas de nascimento distintas, dígito verificador válido).",
              "", "## Testes qui-quadrado (H0: presença de CNS é independente da categoria)", "",
              testes.to_markdown(index=False), ""]
    for c in DIMENSOES:
        sub = tabela1[tabela1["dimensao"] == c]
        linhas += [f"## {c}", "", sub.to_markdown(index=False), ""]
    (OUT_DIR / f"relatorio_tabela1{_sufixo(args.ano_max)}.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"\n📄 {OUT_DIR}/relatorio_tabela1{_sufixo(args.ano_max)}.md")


if __name__ == "__main__":
    main()
