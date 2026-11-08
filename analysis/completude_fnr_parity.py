#!/usr/bin/env python3
"""Etapa 4 do protocolo do estudo: índice de completude cadastral
(adaptado aos campos disponíveis) + precisão/recall/FNR do linkage cego
(`linkage_cego_recall.py`), estratificados por raça/cor, região e tercil de
completude, com IC por bootstrap e teste de paridade de FNR (Equal
Opportunity) entre grupos protegidos.

**Índice de completude — adaptado, não o do protocolo original.** O
protocolo pede nome completo, nome da mãe, data de nascimento completa e
CEP. Conferido nos contratos (`contracts/sia_bi.yml`, `sia_aq.yml`): BI/AQ
**não têm nome nem nome da mãe** — a extração pública do DATASUS suprime
esses campos por privacidade. O índice aqui usa os três componentes
demográficos que EXISTEM em toda linha do BI: sexo informado (`F`/`M`, não
vazio/código genérico), raça/cor informada (não "sem informação"),
município de residência resolvível (não "ignorado/exterior"). Score 0-3
por PACIENTE = o máximo entre os registros dele (a pessoa é bem cadastrada
se qualquer atendimento capturou o dado, mesmo que outro tenha capturado
pior) — computado por `MAX()`/`arg_max()`, determinístico no *score*; o
valor de raça/região usado para estratificar em caso de empate no score
pode variar entre execuções (não-determinismo conhecido do `arg_max`
do DuckDB), o que não afeta a métrica em si.

**Recall/FNR** reusa a decomposição de `linkage_cego_recall.py` (BI×AQ,
blocking por município, classificador +sexo), por PACIENTE em vez de
agregado, para poder estratificar com bootstrap.

**Precisão — nível de bloco, sem produto cartesiano.** "Quantos pares
BI×AQ do mesmo bloco (município) são de fato o mesmo paciente" parece
pedir o produto cartesiano BI×AQ dentro de cada município — que explode
combinatoriamente em cidades grandes (São Paulo: centenas de milhares de
pacientes de cada lado, produto na casa dos bilhões só ali). Em vez disso,
`n_bi` e `n_aq` por bloco vêm de `GROUP BY` separados (nunca materializam
o produto), e `n_true` vem de um JOIN em (cns, município) — quase 1:1, não
combinatorial. Precisão do bloco = n_true / (n_bi × n_aq); a agregada é
Σn_true / Σ(n_bi×n_aq). É a mesma lição do resto desta análise: nunca
materializar um produto cartesiano, mesmo que o resultado final seja
pequeno.

**Paridade de FNR (Equal Opportunity)**: Preto/Pardo vs Branco; Norte/
Nordeste vs Sul/Sudeste. Diferença de FNR com IC por bootstrap (1000
reamostragens); a paridade falha quando o IC da diferença não cobre zero.

Uso:
    python scripts/completude_fnr_parity.py --ano-max 2025

Saídas em `dados/stats/completude_fnr_parity/`:
    metricas_por_estrato.csv     precisão/recall/FNR + IC bootstrap, por estrato
    paridade_fnr.csv             diferença de FNR entre grupos protegidos + IC
    relatorio_completude_fnr.md
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

import _raiz  # noqa: F401
import paths as _paths
from analisar_cns_faltante_bi import RACACOR_MAP, _regiao_uf_case
from validar_cns_bi import digito_verificador_valido

BI_CACHE_DIR = _paths.CACHE / "tabela1_cns_bi"
AQ_CACHE_DIR = _paths.CACHE / "linkage_cego_recall"
OUT_DIR = _paths.STATS / "completude_fnr_parity"
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _sufixo(ano_max: int | None) -> str:
    return "" if ano_max is None else f"_ate{ano_max}"


def _parcial_dir(ano_max: int | None) -> Path:
    return OUT_DIR / f"_parcial{_sufixo(ano_max)}"


def _processar_particao(k: int, memory_limit: str, threads: int, ano_max: int | None) -> None:
    pasta = BI_CACHE_DIR / f"part={k}"
    if not pasta.exists():
        print(f"  partição {k}: pasta BI não existe, pulando")
        return
    aq_path = AQ_CACHE_DIR / f"aq_distinto{_sufixo(ano_max)}.parquet"
    if not aq_path.exists():
        raise SystemExit(f"ERRO: {aq_path} não existe — rode `linkage_cego_recall.py"
                          f"{' --ano-max ' + str(ano_max) if ano_max else ''}` primeiro.")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    filtro_ano = f"AND ano <= {ano_max}" if ano_max is not None else ""
    regiao_case = _regiao_uf_case(_paths.JSONS / "region_state.json", "substr(mun_res, 1, 2)", "regiao")
    raca_linhas = "\n    ".join(f"WHEN racacor = '{cod}' THEN '{lbl}'" for cod, lbl in RACACOR_MAP.items())
    raca_case = f"CASE\n    {raca_linhas}\n    ELSE 'Não informado' END"

    bp = f"read_parquet('{(pasta / '*.parquet').as_posix()}')"
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE _scored AS
      SELECT cns, mun_res, sexo, ano, {raca_case} AS raca_cor, {regiao_case},
        (CASE WHEN sexo IN ('F','M') THEN 1 ELSE 0 END
         + CASE WHEN racacor NOT IN ('99','00') AND racacor IS NOT NULL THEN 1 ELSE 0 END
         + CASE WHEN {regiao_case.split(' AS ')[0]} != 'Ignorado/exterior' THEN 1 ELSE 0 END
        ) AS completude_registro
      FROM {bp}
      WHERE cns_status = 'valido_candidato' {filtro_ano}
    """)

    con.execute("""
      CREATE OR REPLACE TEMP TABLE _melhor AS
      SELECT cns,
             arg_max(raca_cor, completude_registro) AS raca_cor,
             arg_max(regiao, completude_registro) AS regiao,
             arg_max(ano, completude_registro) AS ano,
             MAX(completude_registro) AS completude_score
      FROM _scored
      GROUP BY cns
    """)

    con.execute(f"CREATE OR REPLACE TEMP TABLE _aq AS SELECT * FROM read_parquet('{aq_path.as_posix()}')")

    candidatos = con.execute("SELECT DISTINCT s.cns FROM _scored s INNER JOIN _aq a ON s.cns = a.cns").df()["cns"]
    validos = {c for c in candidatos.unique() if digito_verificador_valido(c)}

    parcial_dir = _parcial_dir(ano_max)
    if not validos:
        pd.DataFrame(columns=["cns", "raca_cor", "regiao", "ano", "completude_score",
                               "tem_blocking", "tem_classificador"]) \
            .to_parquet(parcial_dir / f"pacientes_part={k}.parquet", index=False)
    else:
        validos_df = pd.DataFrame({"cns": list(validos)})
        con.register("_validos_df", validos_df)
        con.execute("CREATE OR REPLACE TEMP TABLE _validos AS SELECT cns FROM _validos_df")

        pacientes = con.execute("""
          SELECT m.cns, m.raca_cor, m.regiao, m.ano, m.completude_score,
            EXISTS (SELECT 1 FROM _scored s JOIN _aq a ON s.cns = a.cns AND s.mun_res = a.mun
                    WHERE s.cns = m.cns) AS tem_blocking,
            EXISTS (SELECT 1 FROM _scored s JOIN _aq a ON s.cns = a.cns AND s.mun_res = a.mun AND s.sexo = a.sexo
                    WHERE s.cns = m.cns) AS tem_classificador
          FROM _melhor m
          INNER JOIN _validos v ON m.cns = v.cns
        """).df()
        pacientes.to_parquet(parcial_dir / f"pacientes_part={k}.parquet", index=False)

    # Precisão por bloco (município) — SEM produto cartesiano: n_bi e n_aq
    # vêm de GROUP BYs separados; n_true vem de um join em (cns, município),
    # quase 1:1, não combinatorial.
    n_bi = con.execute("SELECT mun_res, COUNT(DISTINCT cns) AS n_bi FROM _scored GROUP BY mun_res").df()
    n_aq = con.execute("SELECT mun AS mun_res, COUNT(DISTINCT cns) AS n_aq FROM _aq GROUP BY mun").df()
    n_true = con.execute("""
      SELECT b.mun_res, COUNT(DISTINCT b.cns) AS n_true
      FROM (SELECT DISTINCT mun_res, cns FROM _scored) b
      INNER JOIN (SELECT DISTINCT mun AS mun_res, cns FROM _aq) a
        ON b.cns = a.cns AND b.mun_res = a.mun_res
      GROUP BY b.mun_res
    """).df()
    blocos = n_bi.merge(n_aq, on="mun_res", how="outer").merge(n_true, on="mun_res", how="left").fillna(0)
    blocos.to_parquet(parcial_dir / f"blocos_part={k}.parquet", index=False)

    print(f"  partição {k}: {len(validos):,} pacientes válidos, {len(blocos):,} blocos de município", flush=True)


def _bootstrap_ic(valores: np.ndarray, estimador, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED):
    """IC 95% por bootstrap simples (reamostragem com reposição do vetor de
    valores 0/1). `estimador` recebe um array e devolve um escalar."""
    rng = np.random.default_rng(seed)
    n = len(valores)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ponto = estimador(valores)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        amostra = valores[rng.integers(0, n, n)]
        boots[i] = estimador(amostra)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return ponto, lo, hi


def _precisao_nacional(blocos: pd.DataFrame) -> float:
    """Precisão só existe em nível NACIONAL/de bloco, não por estrato de
    paciente: um bloco (município) mistura pacientes de todas as raças, e o
    lado AQ nem carrega raça — não há como restringir o denominador
    (n_bi × n_aq) a um subgrupo demográfico sem sair do que o bloco
    realmente propõe como candidato. Reportar a mesma precisão nacional em
    toda linha é a leitura honesta; um número "estratificado" aqui seria
    inventado, não medido."""
    soma_true = blocos["n_true"].sum()
    soma_produto = (blocos["n_bi"] * blocos["n_aq"]).sum()
    return soma_true / soma_produto if soma_produto > 0 else float("nan")


def _metricas_estrato(pacientes: pd.DataFrame, precisao_nacional: float, nome_estrato: str) -> dict:
    n = len(pacientes)
    if n == 0:
        return {"estrato": nome_estrato, "n": 0}

    recall = pacientes["tem_classificador"].to_numpy(dtype=float)
    recall_pt, recall_lo, recall_hi = _bootstrap_ic(recall, np.mean)

    fnr = 1 - recall
    fnr_pt, fnr_lo, fnr_hi = _bootstrap_ic(fnr, np.mean)

    return {
        "estrato": nome_estrato, "n": n,
        "recall": recall_pt, "recall_ic_lo": recall_lo, "recall_ic_hi": recall_hi,
        "fnr": fnr_pt, "fnr_ic_lo": fnr_lo, "fnr_ic_hi": fnr_hi,
        "precisao_bloco_nacional": precisao_nacional,
    }


def _teste_paridade_fnr(pacientes: pd.DataFrame, grupo_a: pd.Series, grupo_b: pd.Series,
                         nome: str, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> dict:
    """Diferença de FNR (grupo_a - grupo_b) com IC por bootstrap independente
    em cada grupo. Paridade falha (Equal Opportunity violada) se o IC da
    diferença não cobrir zero."""
    fnr_a = (~pacientes.loc[grupo_a, "tem_classificador"]).to_numpy(dtype=float)
    fnr_b = (~pacientes.loc[grupo_b, "tem_classificador"]).to_numpy(dtype=float)
    if len(fnr_a) == 0 or len(fnr_b) == 0:
        return {"comparacao": nome, "n_a": len(fnr_a), "n_b": len(fnr_b),
                "fnr_a": float("nan"), "fnr_b": float("nan"), "diferenca": float("nan"),
                "ic_lo": float("nan"), "ic_hi": float("nan"), "paridade_rejeitada": None}

    rng = np.random.default_rng(seed)
    na, nb = len(fnr_a), len(fnr_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ma = fnr_a[rng.integers(0, na, na)].mean()
        mb = fnr_b[rng.integers(0, nb, nb)].mean()
        diffs[i] = ma - mb
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {
        "comparacao": nome, "n_a": na, "n_b": nb,
        "fnr_a": fnr_a.mean(), "fnr_b": fnr_b.mean(), "diferenca": fnr_a.mean() - fnr_b.mean(),
        "ic_lo": lo, "ic_hi": hi, "paridade_rejeitada": not (lo <= 0 <= hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npart", type=int, default=24)
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ano-max", type=int, default=None)
    ap.add_argument("--processar-particao", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parcial_dir = _parcial_dir(args.ano_max)
    parcial_dir.mkdir(parents=True, exist_ok=True)

    if args.processar_particao is not None:
        _processar_particao(args.processar_particao, args.memory_limit, args.threads, args.ano_max)
        return

    if not BI_CACHE_DIR.exists():
        raise SystemExit(f"ERRO: {BI_CACHE_DIR} não existe — rode `tabela1_cns_nacional.py` primeiro.")

    print("Fase 1 — por partição: completude, blocking/classificador, blocos de precisão")
    for k in range(args.npart):
        if (parcial_dir / f"pacientes_part={k}.parquet").exists():
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

    print("\nFase 2 — agregação nacional")
    pacientes = pd.concat([pd.read_parquet(f) for f in sorted(parcial_dir.glob("pacientes_part=*.parquet"))],
                          ignore_index=True)
    blocos = pd.concat([pd.read_parquet(f) for f in sorted(parcial_dir.glob("blocos_part=*.parquet"))],
                       ignore_index=True)
    blocos = blocos.groupby("mun_res", as_index=False)[["n_bi", "n_aq", "n_true"]].sum()

    # Tercis por QUANTIL não funcionam aqui: o escore é discreto (0-3) e
    # 95%+ dos pacientes cravam no valor máximo (3), então quantile([1/3,2/3])
    # devolve [3,3] e todo mundo cai no mesmo grupo. Estratifica pelo valor discreto
    # do escore mesmo (1/2/3 — 0 não ocorre nesta amostra), que é honesto
    # E mais interpretável (\"escore=X\" já diz quantos dos 3 campos batem).
    pacientes["tercil_completude"] = pacientes["completude_score"].map(
        lambda s: f"Escore {s}/3")

    print(f"Total de pacientes válidos: {len(pacientes):,}")
    print(f"Distribuição do escore de completude: {pacientes['completude_score'].value_counts().sort_index().to_dict()}")

    precisao_nacional = _precisao_nacional(blocos)
    linhas = []
    linhas.append(_metricas_estrato(pacientes, precisao_nacional, "Nacional"))
    for raca, sub in pacientes.groupby("raca_cor"):
        linhas.append(_metricas_estrato(sub, precisao_nacional, f"Raça/cor: {raca}"))
    for regiao, sub in pacientes.groupby("regiao"):
        linhas.append(_metricas_estrato(sub, precisao_nacional, f"Região: {regiao}"))
    for terc, sub in pacientes.groupby("tercil_completude"):
        linhas.append(_metricas_estrato(sub, precisao_nacional, f"Completude: {terc}"))

    metricas = pd.DataFrame(linhas)
    metricas.to_csv(OUT_DIR / f"metricas_por_estrato{_sufixo(args.ano_max)}.csv", index=False)

    # O ano é confundidor mais forte que a raça (V = 0,324 vs 0,197) na leitura
    # completude -> recall. Corte em 2 faixas pelo ANO MEDIANO da amostra
    # (não um ponto de corte chutado) -- within cada faixa, o escore de
    # completude é reestratificado; se o padrão (recall despenca de 3/3
    # para 2/3) sobreviver nas duas faixas, completude não é só um proxy
    # de ano.
    if "ano" in pacientes.columns and pacientes["ano"].notna().any():
        corte = int(pacientes["ano"].median())
        pacientes["faixa_ano"] = np.where(pacientes["ano"] <= corte,
                                           f"até {corte}", f"{corte+1}+")
        print(f"\nCorte de ano (mediana): até {corte} vs {corte+1}+")
        print(f"Distribuição: {pacientes['faixa_ano'].value_counts().to_dict()}")

        linhas_ano = []
        for faixa, sub_faixa in pacientes.groupby("faixa_ano"):
            linhas_ano.append(_metricas_estrato(sub_faixa, precisao_nacional, f"Ano: {faixa}"))
            for terc, sub in sub_faixa.groupby("tercil_completude"):
                linhas_ano.append(_metricas_estrato(
                    sub, precisao_nacional, f"Ano: {faixa} | Completude: {terc}"))
        metricas_ano = pd.DataFrame(linhas_ano)
        metricas_ano.to_csv(OUT_DIR / f"metricas_ano_x_completude{_sufixo(args.ano_max)}.csv", index=False)
        print(f"\n📄 {OUT_DIR}/metricas_ano_x_completude{_sufixo(args.ano_max)}.csv")

    # Estratificação CONJUNTA raça × completude — o achado de completude
    # (0% -> 86% -> 98% de recall) é bem maior que o de raça sozinha (≤1pp
    # de diferença nacional); um peso IPW que varie só por raça mal se move.
    # Reusa os dados de paciente já em cache (nenhuma varredura nova).
    linhas_conjunta = []
    for (raca, esc), sub in pacientes.groupby(["raca_cor", "completude_score"]):
        linhas_conjunta.append({**_metricas_estrato(sub, precisao_nacional, f"{raca} × escore {esc}/3"),
                                 "raca_cor": raca, "completude_score": esc})
    conjunta = pd.DataFrame(linhas_conjunta)
    conjunta.to_csv(OUT_DIR / f"metricas_raca_x_completude{_sufixo(args.ano_max)}.csv", index=False)
    print("\n--- raça × completude (célula com n<500 é ruído, use com cautela) ---")
    print(conjunta[["estrato", "n", "recall", "fnr"]].to_string(index=False))

    preto_pardo = pacientes["raca_cor"].isin(["Preta", "Parda"])
    branco = pacientes["raca_cor"] == "Branca"
    norte_nordeste = pacientes["regiao"].isin(["Norte", "Nordeste"])
    sul_sudeste = pacientes["regiao"].isin(["Sul", "Sudeste"])

    paridades = pd.DataFrame([
        _teste_paridade_fnr(pacientes, preto_pardo, branco, "Preto/Pardo vs Branco"),
        _teste_paridade_fnr(pacientes, norte_nordeste, sul_sudeste, "Norte/Nordeste vs Sul/Sudeste"),
    ])
    paridades.to_csv(OUT_DIR / f"paridade_fnr{_sufixo(args.ano_max)}.csv", index=False)

    print("\n" + metricas.to_string(index=False))
    print("\n" + paridades.to_string(index=False))

    linhas_md = [
        "# Completude cadastral e paridade de FNR (BI × AQ)", "",
        f"**{len(pacientes):,} pacientes** válidos (CNS com dígito verificador correto, presente em BI e AQ). "
        f"Escore de completude 0-3 (sexo + raça/cor + região informados), estratificado pelo valor "
        f"discreto (não por tercil — a distribuição está muito concentrada no máximo para terciar): "
        f"{pacientes['completude_score'].value_counts().sort_index().to_dict()}.", "",
        "## Métricas por estrato (recall/FNR com IC 95% por bootstrap; precisão no nível de bloco)", "",
        metricas.to_markdown(index=False), "",
        "## Paridade de FNR (Equal Opportunity)", "",
        paridades.to_markdown(index=False), "",
        "`paridade_rejeitada = True` quando o IC de 95% da diferença de FNR entre os grupos NÃO cobre zero "
        "— ou seja, a taxa de falso negativo do linkage é estatisticamente diferente entre os grupos.",
    ]
    (OUT_DIR / f"relatorio_completude_fnr{_sufixo(args.ano_max)}.md").write_text("\n".join(linhas_md), encoding="utf-8")
    print(f"\n📄 {OUT_DIR}/relatorio_completude_fnr{_sufixo(args.ano_max)}.md")


if __name__ == "__main__":
    main()
