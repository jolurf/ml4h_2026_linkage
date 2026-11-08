#!/usr/bin/env python3
"""Recall interno das chaves REAIS da cascata (R1/R3/R4, que exigem data de
nascimento exata) dentro do próprio BI, que tem data de nascimento — em vez
de bater BI×AQ com uma chave substituta.

**Isto é DIFERENTE de `cascade_bi_estabilidade.py`.** Aquele mede
ESTABILIDADE: exige que a chave seja IGUAL em TODOS os registros do
paciente (`COUNT(DISTINCT chave) == 1`) — um teste mais rígido do que o
blocking real precisa, porque blocking só precisa de UM registro do lado
do "pool" que bata com a chave do lado da "consulta", não que todos os
registros do paciente concordem entre si. Por isso o paper já chama a
estabilidade de "lower bound on loss, not an estimate of it": ela é
pessimista demais.

Este script mede RECALL, não estabilidade: para cada registro de um
paciente com ≥2 registros válidos no BI, a pergunta é "se ESTE registro
fosse escondido e os demais do mesmo paciente formassem o pool de
blocking, pelo menos um bateria?" -- equivalente a marginalizar sobre
QUAL registro seria o escondido, em vez de fixar um só por desempate
(reter um registro fixo por ROW_NUMBER exigiria funções de janela que
estouram a memória contra a partição inteira; marginalizar é mais barato E estatisticamente
mais correto, porque não depende de uma escolha arbitrária de qual
registro é "a consulta"). Recall = fração dos REGISTROS cujo valor não é
único dentro do próprio paciente. Isso é estritamente mais informativo
que estabilidade (todo paciente estável tem 100% dos registros
não-singleton; o inverso não) e é a quantidade que corresponde de
verdade a "blocking encontraria o candidato certo".

**O que isto NÃO resolve, e por quê é importante dizer isso**: tanto este
teste quanto o de estabilidade continuam DENTRO do BI — mesmo sistema,
mesma rede de estabelecimentos ambulatoriais, mesmo fluxo de cadastro.
O problema real que a cascata SIA→SIH enfrenta cruza para um sistema
ESTRUTURALMENTE diferente (internação, outra rede de
estabelecimentos — o CNES do ambulatório de rotina de um paciente não
precisa ser o CNES do hospital onde ele é internado). Testar só dentro
do BI mede continuidade de cuidado ambulatorial, que é uma coisa mais
fácil que concordância ambulatório↔hospital — então este número tende a
SUPERESTIMAR o recall real da cascata SIA→SIH, do mesmo jeito que a
estabilidade tende a SUBESTIMAR (por exigir que todos os registros
concordem). Os dois juntos dão um intervalo mais honesto do que qualquer
um sozinho.

**Ambiguidade (colisão com outro paciente) não entra aqui** — medi-la via
`COUNT(DISTINCT cns)` em janela sobre a partição inteira estoura a memória
(6 GB). A precisão, com colisão entre pacientes, é medida em
`cascata_registro_retido_uf.py`.

Reusa o cache já materializado por `cascade_bi_estabilidade.py`
(`dados/cache/cascade_bi_estabilidade_ate2025/`) — não refaz a
varredura do BI bruto.

Uso:
    python scripts/cascade_bi_recall_interno.py --ano-max 2025

Saídas em `dados/stats/cascade_bi_recall_interno/`:
    recall_chaves_ate2025.csv
    relatorio_recall_interno_ate2025.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths
from validar_cns_bi import digito_verificador_valido

OUT_DIR = _paths.STATS / "cascade_bi_recall_interno"


def _sufixo(ano_max: int | None) -> str:
    return "" if ano_max is None else f"_ate{ano_max}"


def _cache_dir(ano_max: int | None) -> Path:
    # Reusa o cache de cascade_bi_estabilidade.py -- mesmas colunas
    # (cns, dtnasc, sexo, mun_res, mun_ubs, cnes, cns_status, part).
    return _paths.CACHE / f"cascade_bi_estabilidade{_sufixo(ano_max)}"


def _parcial_dir(ano_max: int | None) -> Path:
    return OUT_DIR / f"_parcial{_sufixo(ano_max)}"


def _processar_particao(k: int, memory_limit: str, threads: int, ano_max: int | None) -> None:
    pasta = _cache_dir(ano_max) / f"part={k}"
    if not pasta.exists():
        print(f"  partição {k}: pasta do cache não existe, pulando")
        return

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    bp = f"read_parquet('{(pasta / '*.parquet').as_posix()}')"

    # Funcoes de janela estouraram memoria (6GB) mesmo so com COUNT(*) --
    # 4 formas de PARTITION BY diferentes na mesma passada forcam DuckDB a
    # materializar varias ordenacoes/hashes simultaneos sobre a particao
    # inteira. O script de estabilidade (que funciona) faz tudo num GROUP
    # BY cns so; aqui uso o mesmo padrao, em duas etapas.
    #
    # Redefinicao de recall que elimina a necessidade de "reter um
    # registro especifico": em vez de fixar por desempate QUAL registro e
    # a "consulta" (arbitrario), conto, por chave, quantos registros tem
    # um valor que NAO se repete em nenhum outro registro do mesmo
    # paciente ("singleton") -- um desses, se retido, faria blocking
    # falhar; qualquer outro, se retido, blocking acha. Recall nacional =
    # 1 - (registros singleton / registros totais), marginalizando sobre
    # QUAL registro seria o retido em vez de fixar um só -- mais barato
    # de computar E mais correto estatisticamente que a escolha
    # deterministica original.
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE _freq AS
      SELECT cns, dtnasc, cnes, sexo, mun_res, mun_ubs,
        COUNT(*) AS f_r1
      FROM {bp}
      WHERE cns_status = 'valido_candidato'
      GROUP BY cns, dtnasc, cnes, sexo, mun_res, mun_ubs
    """)
    # f_r1 acima e por tupla completa (todos os campos) so pra contar
    # registros fisicos distintos; as frequencias REAIS por chave vem de
    # agregar de novo, agora so pelos campos daquela chave.
    con.execute("""
      CREATE OR REPLACE TEMP TABLE _por_r1 AS
      SELECT cns, dtnasc, cnes, SUM(f_r1) AS n FROM _freq GROUP BY cns, dtnasc, cnes
    """)
    con.execute("""
      CREATE OR REPLACE TEMP TABLE _por_r3 AS
      SELECT cns, dtnasc, sexo, mun_res, SUM(f_r1) AS n FROM _freq GROUP BY cns, dtnasc, sexo, mun_res
    """)
    con.execute("""
      CREATE OR REPLACE TEMP TABLE _por_r4 AS
      SELECT cns, dtnasc, sexo, mun_ubs, SUM(f_r1) AS n FROM _freq GROUP BY cns, dtnasc, sexo, mun_ubs
    """)

    candidatos_df = con.execute("""
      SELECT cns, SUM(f_r1) AS n_registros FROM _freq GROUP BY cns HAVING SUM(f_r1) >= 2
    """).df()
    validos = {c for c in candidatos_df["cns"].unique() if digito_verificador_valido(c)}

    if not validos:
        pd.DataFrame(columns=["n_registros_total", "n_r1_singleton", "n_r3_singleton", "n_r4_singleton"]) \
            .to_parquet(_parcial_dir(ano_max) / f"part={k}.parquet", index=False)
        print(f"  partição {k}: 0 pacientes válidos com ≥2 registros")
        return

    validos_df = pd.DataFrame({"cns": list(validos)})
    con.register("_validos_df", validos_df)
    con.execute("CREATE OR REPLACE TEMP TABLE _validos AS SELECT cns FROM _validos_df")

    def _somar_registros_e_singletons(tabela: str) -> tuple[int, int]:
        r = con.execute(f"""
          SELECT SUM(t.n) AS total,
                 SUM(CASE WHEN t.n = 1 THEN 1 ELSE 0 END) AS singletons
          FROM {tabela} t INNER JOIN _validos v ON v.cns = t.cns
        """).fetchone()
        return int(r[0] or 0), int(r[1] or 0)

    tot_r1, sing_r1 = _somar_registros_e_singletons("_por_r1")
    tot_r3, sing_r3 = _somar_registros_e_singletons("_por_r3")
    tot_r4, sing_r4 = _somar_registros_e_singletons("_por_r4")
    # tot_r1/tot_r3/tot_r4 derivam da MESMA soma (total de registros dos
    # pacientes validos) por tres agregacoes diferentes -- tem que bater;
    # serve de checagem de consistencia, nao so denominador.
    assert tot_r1 == tot_r3 == tot_r4, f"denominadores divergem: {tot_r1=} {tot_r3=} {tot_r4=}"
    n_registros_total = tot_r1

    resumo = pd.DataFrame([{
        "n_registros_total": n_registros_total,
        "n_r1_singleton": sing_r1,
        "n_r3_singleton": sing_r3,
        "n_r4_singleton": sing_r4,
    }])
    resumo.to_parquet(_parcial_dir(ano_max) / f"part={k}.parquet", index=False)
    print(f"  partição {k}: {len(validos):,} pacientes (≥2 registros), {n_registros_total:,} registros; "
          f"singleton R1 {sing_r1:,}, R3 {sing_r3:,}, R4 {sing_r4:,}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npart", type=int, default=24)
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ano-max", type=int, default=None)
    ap.add_argument("--processar-particao", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not _cache_dir(args.ano_max).exists():
        raise SystemExit(f"ERRO: {_cache_dir(args.ano_max)} não existe — rode "
                          f"`cascade_bi_estabilidade.py --ano-max {args.ano_max}` primeiro "
                          f"(este script reusa aquele cache).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parcial_dir = _parcial_dir(args.ano_max)
    parcial_dir.mkdir(parents=True, exist_ok=True)

    if args.processar_particao is not None:
        _processar_particao(args.processar_particao, args.memory_limit, args.threads, args.ano_max)
        return

    print("Recall interno (BI, chaves reais R1/R3/R4) — por partição")
    for k in range(args.npart):
        if (parcial_dir / f"part={k}.parquet").exists():
            print(f"  partição {k+1}/{args.npart} já concluída (resumindo) — pulando")
            continue
        t0 = time.time()
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--processar-particao", str(k),
               "--memory-limit", args.memory_limit, "--threads", str(args.threads)]
        if args.ano_max is not None:
            cmd += ["--ano-max", str(args.ano_max)]
        subprocess.run(cmd, check=True)
        print(f"  partição {k+1}/{args.npart} ok ({time.time()-t0:.0f}s)", flush=True)

    print("\nAgregação nacional")
    partes = [pd.read_parquet(f) for f in sorted(parcial_dir.glob("part=*.parquet"))]
    tot = pd.concat(partes, ignore_index=True).sum()
    n_registros = int(tot["n_registros_total"])

    linhas_resumo = []
    for chave, col_sing in [
        ("R1 (nasc+CNES)", "n_r1_singleton"),
        ("R3 (nasc+sexo+mun_res)", "n_r3_singleton"),
        ("R4 (nasc+sexo+mun_ubs)", "n_r4_singleton"),
    ]:
        n_sing = int(tot[col_sing])
        n_recall = n_registros - n_sing
        taxa = n_recall / n_registros if n_registros else float("nan")
        linhas_resumo.append({
            "chave": chave, "n_registros": n_registros, "n_recall": n_recall, "taxa_recall": taxa,
        })
        print(f"{chave}: recall {n_recall:,}/{n_registros:,} registros ({100*taxa:.2f}%)")

    resumo = pd.DataFrame(linhas_resumo)
    resumo.to_csv(OUT_DIR / f"recall_chaves{_sufixo(args.ano_max)}.csv", index=False)

    # compara com a estabilidade (mesma populacao, mesmo cache) se o
    # arquivo ja existir -- a leitura esperada e n_recall >= n_estavel
    # sempre, porque recall e um teste mais fraco que estabilidade.
    estab_path = _paths.STATS / "cascade_bi_estabilidade" / f"estabilidade_chaves{_sufixo(args.ano_max)}.csv"
    comparacao = ""
    if estab_path.exists():
        estab = pd.read_csv(estab_path).set_index("chave")
        linhas_comp = ["", "## Comparação com estabilidade (mesmo cache, mesma população)", "",
                       "| Chave | Estabilidade (todos os registros concordam) | Recall interno (um registro acha o resto) |",
                       "|---|---|---|"]
        for chave in resumo["chave"]:
            if chave in estab.index:
                t_estab = estab.loc[chave, "taxa_estabilidade"]
                t_recall = resumo.set_index("chave").loc[chave, "taxa_recall"]
                linhas_comp.append(f"| {chave} | {100*t_estab:.2f}% | {100*t_recall:.2f}% |")
        comparacao = "\n".join(linhas_comp)

    titulo_janela = f" (até {args.ano_max})" if args.ano_max else ""
    linhas = [
        f"# Recall interno das chaves R1/R3/R4 dentro do BI{titulo_janela}", "",
        f"Registros de pacientes com ≥2 registros válidos: **{n_registros:,}**.", "",
        resumo.to_markdown(index=False), "",
        "Recall = fração dos registros cujo valor da chave NÃO é único dentro do próprio "
        "paciente -- ou seja, se esse registro específico fosse retido (escondido) e os "
        "demais do mesmo paciente formassem o pool de blocking, pelo menos um deles ainda "
        "bateria. Equivalente a marginalizar sobre qual registro seria o retido (em vez de "
        "fixar um só por desempate), o que é mais barato de computar e mais correto "
        "estatisticamente. Mais fraco, e mais realista do blocking "
        "de verdade, do que a estabilidade (que exige TODOS os registros concordarem). "
        "Não mede ambiguidade (colisão com outro paciente): COUNT(DISTINCT cns) em "
        "janela sobre a partição inteira estoura a memória (6 GB); a precisão com "
        "colisão é medida em cascata_registro_retido_uf.py. "
        "Mede só dentro do BI (mesmo sistema ambulatorial): não captura a "
        "heterogeneidade de cruzar para um sistema estruturalmente diferente (internação), "
        "que é o problema real da cascata SIA→SIH -- este número tende a SUPERESTIMAR o "
        "recall real dessa cascata, do mesmo jeito que a estabilidade tende a SUBESTIMAR.",
        comparacao,
    ]
    (OUT_DIR / f"relatorio_recall_interno{_sufixo(args.ano_max)}.md").write_text(
        "\n".join(linhas), encoding="utf-8")
    print(f"\n📄 {OUT_DIR}/relatorio_recall_interno{_sufixo(args.ano_max)}.md")


if __name__ == "__main__":
    main()
