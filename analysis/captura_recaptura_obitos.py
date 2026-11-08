#!/usr/bin/env python3
"""Quantos óbitos nenhuma fonte registrou, e por que as fontes discordam,
incluindo os limites do link com o registro de câncer.

A FNR do linkage cego BI×AQ (~2–4%) é perda DENTRO do SIA. O passo que leva o
óbito do SIM ao paciente (Camada 2 do `2_join_final_sim`) tem filtros próprios —
chave exata nascimento + sexo + município de residência NO ÓBITO, óbito depois do
último registro no SUS, e um CID do paciente igual (`list_contains`, igualdade
exata) a `CAUSABAS` ou a uma linha da declaração — e as fontes discordam muito
mais que 3%: dentro de 5 anos, pipeline e RHC concordam em 229 óbitos, 111 só no
RHC, 134 só no pipeline. Três medidas:

(a) **Captura-recaptura** (Chapman, variância de Seber) sobre as duas listas,
    no total e estratificada por estádio, tempo até o óbito e ano. Lincoln-
    Petersen assume listas independentes e população fechada: dependência
    positiva (o paciente bem registrado aparece nas duas) SUBESTIMA o total;
    vínculos falsos numa lista o SUPERESTIMAM. Por isso (c).
(b) **Óbitos faltantes na referência**: os N̂ − 474 óbitos que nenhuma lista
    registrou são acrescentados entre sobreviventes (1 000 sorteios, tempo
    sorteado dos óbitos observados, limitado ao seguimento), ao acaso e
    estratificados por estádio pela captura-recaptura do estrato.
(c) **Decomposição dos óbitos só-RHC**: busca no SIM pela chave data do óbito +
    nascimento (a de `buscar_causa_sim.py`, sem município e sem CID), validada
    nos óbitos que as duas fontes já têm. Classifica cada um pelo filtro do
    pipeline que o perdeu, ou como ausente do SIM.
(d) Quantos dos só-RHC entraram na coorte pela rota de município do link RHC.
(e) **Os dois erros, juntos.** Entre os óbitos que as duas fontes têm, a chave
    não acha ~15% no SIM; entre os só-RHC, muito mais. O excesso é tratado como
    vínculo RHC↔SUS falso (ou data errada): sai da lista do RHC numa captura-
    recaptura ajustada, e é censurado num cenário que ao mesmo tempo acrescenta
    os faltantes ajustados. Os dois erros puxam a OS em sentidos opostos.

Saídas em `experimento/` (agregadas, copiadas para outputs/):
    captura_recaptura_obitos.csv, captura_recaptura_cenarios.csv, decomposicao_so_rhc.csv
O detalhe por paciente fica só em `experimento/decomposicao_so_rhc_pacientes.csv`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from propagacao_erro_linkage import acrescentar_obitos, carregar, medir

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
EXP = AQUI / "experimento"
AGREGADOS = RAIZ / "outputs"
DO = RAIZ / "dados" / "parquets" / "DO"
SEMENTE = 20261009
N_SORTEIOS = 1000
ESTADIOS = {"I–II": ["I-Initial", "II-Local"],
            "III–V": ["III-Regional", "IV-Metastatic", "V-Bilateral"],
            "Ignorado": ["Ignored"]}


def chapman(a: int, b: int, c: int) -> dict:
    """a = nas duas listas, b = só RHC, c = só pipeline."""
    n1, n2 = a + b, a + c
    N = (n1 + 1) * (n2 + 1) / (a + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - a) * (n2 - a) / ((a + 1) ** 2 * (a + 2))
    lo, hi = N - 1.96 * var ** .5, N + 1.96 * var ** .5
    obs = a + b + c
    return {"ambos": a, "so_rhc": b, "so_pipeline": c, "observados": obs,
            "N_lincoln_petersen": n1 * n2 / a if a else np.nan,
            "N_chapman": N, "N_ic_lo": lo, "N_ic_hi": hi,
            "faltantes": N - obs, "faltantes_ic_lo": lo - obs, "faltantes_ic_hi": hi - obs,
            "captura_pipeline": n2 / N, "captura_rhc": n1 / N, "captura_uniao": obs / N}


def listas(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ev = d["E_final"] == 1
    return (d["evento_obito_inca"] == 1) & ev, (d["evento_obito_sus"] == 1) & ev


def captura_recaptura(d: pd.DataFrame) -> pd.DataFrame:
    R, P = listas(d)
    estratos = {"total": {"total": pd.Series(True, index=d.index)},
                "estadio": {k: d["cat_estadiamento"].isin(v) for k, v in ESTADIOS.items()},
                "tempo_ate_obito": {"≤ 12 meses": d["T_meses"] <= 12, "12–60 meses": d["T_meses"] > 12},
                "ano_diagnostico": {"≤ 2015": d["ANO_DIAGN"] <= 2015, "> 2015": d["ANO_DIAGN"] > 2015}}
    linhas = []
    for eixo, grupos in estratos.items():
        for g, m in grupos.items():
            linhas.append({"eixo": eixo, "estrato": g,
                           **chapman(int((R & P & m).sum()), int((R & ~P & m).sum()), int((~R & P & m).sum()))})
        if eixo != "total":
            parc = [r for r in linhas if r["eixo"] == eixo]
            linhas.append({"eixo": eixo, "estrato": "soma dos estratos",
                           "observados": sum(r["observados"] for r in parc),
                           "N_chapman": sum(r["N_chapman"] for r in parc),
                           "faltantes": sum(r["faltantes"] for r in parc)})
    t = pd.DataFrame(linhas)
    tot = t.iloc[0]
    assert (tot.ambos, tot.so_rhc, tot.so_pipeline) == (229, 111, 134), tot
    assert abs(tot.N_chapman - 538.7) < 0.05, tot.N_chapman
    return t


def ajuste_falsos(dec: pd.DataFrame) -> dict:
    """Quantos óbitos só-RHC são provavelmente falsos (de outro paciente).

    Com F falsos entre N só-RHC e taxa de perda da chave b (não-encontrados entre
    os 'ambos'), os não-encontrados são F + b·(N − F); logo F = (não − b·N)/(1 − b).
    Subtrair b·N sem dividir por (1 − b) aplicava a perda da chave também aos
    falsos e subestimava F. É um LIMITE SUPERIOR: b foi medido em óbitos que o
    link já tinha achado, mais fáceis de achar que um óbito verdadeiro perdido."""
    g = dec.set_index(["grupo", "classe"])["n"]
    nao = "não encontrado no SIM (provável vínculo RHC↔SUS falso ou data divergente)"
    base = g.get(("ambos", nao), 0) / g[("ambos", "TOTAL")]
    so_nao = g.get(("so_rhc", nao), 0)
    excesso = max(0.0, (so_nao - base * g[("so_rhc", "TOTAL")]) / (1 - base))
    return {"taxa_nao_encontrado_base": base, "so_rhc_nao_encontrados": int(so_nao),
            "so_rhc_falsos_estimados": excesso}


def cenarios(d: pd.DataFrame, cr: pd.DataFrame, pac: pd.DataFrame, aj: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    tempos = d.loc[d["E_final"] == 1, "T_meses"].to_numpy()
    E0 = d["E_final"].to_numpy().astype(int)
    ref = medir(d, E0, com_hr=True)
    assert abs(ref["os5"] - 80.49) < 0.05 and abs(ref["hr"] - 2.61) < 0.01, ref
    tot = cr.iloc[0]
    est = cr[(cr.eixo == "estadio") & (cr.estrato != "soma dos estratos")].set_index("estrato")
    linhas = [{"cenario": "referência", "k_total": 0, "n_sorteios": 1, "os5_med": ref["os5"],
               "hr_med": ref["hr"], "gap_med": ref["gap"]}]

    def rodar(nome: str, ks: dict) -> None:
        rows = []
        for i in range(N_SORTEIOS):
            E, T = E0.copy(), None
            for rot, k in ks.items():
                m = None if rot == "todos" else d["cat_estadiamento"].isin(ESTADIOS[rot]).to_numpy()
                tp = tempos if m is None else d.loc[(d["E_final"] == 1) & m, "T_meses"].to_numpy()
                E, T = acrescentar_obitos(d, E, round(max(k, 0)), tp, rng, elegiveis=m, T=T)
            rows.append(medir(d, E, T, com_hr=i < 200))
        r = pd.DataFrame(rows)
        linhas.append({"cenario": nome, "k_total": int(sum(round(max(k, 0)) for k in ks.values())),
                       "n_sorteios": len(r), "os5_med": r.os5.median(), "os5_p025": r.os5.quantile(.025),
                       "os5_p975": r.os5.quantile(.975), "hr_med": r.hr.median(),
                       "hr_p025": r.hr.quantile(.025), "hr_p975": r.hr.quantile(.975),
                       "gap_med": r.gap.median(), "gap_p025": r.gap.quantile(.025), "gap_p975": r.gap.quantile(.975)})
        print(f"  {nome}: ok", flush=True)

    # óbitos só-RHC não encontrados no SIM, além da linha de base: candidatos a vínculo falso
    nao = pac.loc[(pac.grupo == "so_rhc") & ~pac.achou, "id_paciente"]
    idx_nao = np.where(d["id_paciente"].isin(set(nao)).to_numpy())[0]
    k_fals = round(aj["so_rhc_falsos_estimados"])
    tot_aj = cr[cr.estrato == "total, sem os prováveis falsos do RHC"].iloc[0]

    def rodar_misto(nome: str, k_remover: int, k_acrescentar: float) -> None:
        rows = []
        for i in range(N_SORTEIOS):
            E = E0.copy()
            E[rng.choice(idx_nao, size=k_remover, replace=False)] = 0  # censura no mesmo T
            E, T = acrescentar_obitos(d, E, round(max(k_acrescentar, 0)), tempos, rng)
            rows.append(medir(d, E, T, com_hr=i < 200))
        r = pd.DataFrame(rows)
        linhas.append({"cenario": nome, "k_total": int(round(k_acrescentar)) - k_remover, "n_sorteios": len(r),
                       "os5_med": r.os5.median(), "os5_p025": r.os5.quantile(.025), "os5_p975": r.os5.quantile(.975),
                       "hr_med": r.hr.median(), "hr_p025": r.hr.quantile(.025), "hr_p975": r.hr.quantile(.975),
                       "gap_med": r.gap.median(), "gap_p025": r.gap.quantile(.025), "gap_p975": r.gap.quantile(.975)})
        print(f"  {nome}: ok", flush=True)

    rodar("faltantes por estádio (principal)", {g: est.loc[g, "faltantes"] for g in ESTADIOS})
    rodar("faltantes ao acaso", {"todos": tot.faltantes})
    rodar("faltantes ao acaso, limite inferior do IC", {"todos": tot.faltantes_ic_lo})
    rodar("faltantes ao acaso, limite superior do IC", {"todos": tot.faltantes_ic_hi})
    rodar_misto("só remover os prováveis falsos do RHC", k_fals, 0)
    rodar_misto("remover os prováveis falsos do RHC e acrescentar os faltantes ajustados",
                k_fals, tot_aj.faltantes)
    return pd.DataFrame(linhas)


def decomposicao(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Busca no SIM, por data do óbito + nascimento, os óbitos só-RHC (e os que as
    duas fontes têm, como validação da chave)."""
    R, P = listas(d)
    so_rhc, ambos = R & ~P, R & P
    alvo = d[so_rhc | ambos].copy()
    alvo["grupo"] = np.where(so_rhc[alvo.index], "so_rhc", "ambos")
    alvo["dtobito"] = pd.to_datetime(alvo["DATAOBITO"]).dt.strftime("%d%m%Y")
    alvo["dtnasc"] = pd.to_datetime(alvo["nascimento"]).dt.strftime("%d%m%Y")
    alvo["sexo_pac"] = pd.to_numeric(alvo["SEXO"], errors="coerce").astype("Int64").astype(str)
    alvo["mun_pac"] = alvo["mun_res_final"].astype(str).str.split(".").str[0].str.zfill(6).str[:6]
    # CIDs do paciente no SUS, como a Camada 2 os usa (`lista_cids` = cid_principal distintos)
    cids = (pd.read_parquet(AQUI / "df_final_scored.parquet", columns=["id_unico", "cid_principal"])
            .dropna().astype(str).groupby("id_unico")["cid_principal"].apply(lambda s: sorted(set(s))))
    alvo["cids"] = alvo["id_paciente"].map(cids)

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.register("a", alvo[["id_paciente", "dtobito", "dtnasc"]])
    sim = con.execute(f"""
        SELECT a.id_paciente, s.SEXO AS sexo_sim, SUBSTR(CAST(TRY_CAST(s.CODMUNRES AS BIGINT) AS VARCHAR), 1, 6) AS mun_sim,
               s.CAUSABAS, s.LINHAA, s.LINHAB, s.LINHAC, s.LINHAD, s.LINHAII
        FROM a JOIN read_parquet('{DO}/**/*.parquet', hive_partitioning=true, union_by_name=true) s
          ON s.DTOBITO = a.dtobito AND s.DTNASC = a.dtnasc""").df()
    n_cand = sim.groupby("id_paciente").size()
    m = alvo.merge(sim, on="id_paciente", how="left")
    campos = ["CAUSABAS", "LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII"]

    def cid_exato(r) -> bool:  # a regra do pipeline: igualdade exata
        cs = set(r["cids"]) if isinstance(r["cids"], list) else set()
        return any(str(r[c]) in cs for c in campos if pd.notna(r[c]))

    def cid_contido(r) -> bool:  # leitura generosa: o código aparece em algum campo
        cs = r["cids"] if isinstance(r["cids"], list) else []
        txt = " ".join(str(r[c]) for c in campos if pd.notna(r[c]))
        return any(c[:3] in txt for c in cs)
    achou = m["CAUSABAS"].notna()
    m["confere_sexo"] = achou & (m["sexo_sim"].astype(str) == m["sexo_pac"])
    m["confere_mun"] = achou & (m["mun_sim"] == m["mun_pac"])
    m["cid_exato"] = achou & m.apply(cid_exato, axis=1)
    m["cid_contido"] = achou & m.apply(cid_contido, axis=1)
    m["causa_c64"] = achou & m["CAUSABAS"].astype(str).str.startswith(("C64", "C65"))
    m["n_candidatos"] = m["id_paciente"].map(n_cand).fillna(0).astype(int)
    p = m.groupby(["id_paciente", "grupo"]).agg(
        achou=("CAUSABAS", lambda s: s.notna().any()), ambiguo=("n_candidatos", lambda s: s.max() > 1),
        confere_sexo=("confere_sexo", "max"), confere_mun=("confere_mun", "max"),
        cid_exato=("cid_exato", "max"), cid_contido=("cid_contido", "max"),
        causa_c64=("causa_c64", "max"), rota_mun=("TIPO_MATCH", lambda s: (s == "MUNICIPIO_RESIDENCIA").any())
    ).reset_index()

    def classe(r) -> str:
        if not r.achou:
            return "não encontrado no SIM (provável vínculo RHC↔SUS falso ou data divergente)"
        if r.ambiguo:
            return "encontrado, chave ambígua"
        if not r.confere_mun:
            return "encontrado; município de residência difere (bloqueio por município)"
        if not r.confere_sexo:
            return "encontrado; sexo difere"
        if not r.cid_exato:
            return "encontrado; nenhum CID do paciente igual a um campo da DO (filtro semântico)"
        return "encontrado; passa nos três filtros (óbito antes do último registro no SUS, ou outra)"
    p["classe"] = p.apply(classe, axis=1)
    linhas = []
    for g, sub in p.groupby("grupo"):
        for c, n in sub["classe"].value_counts().items():
            linhas.append({"grupo": g, "classe": c, "n": int(n), "pct": 100 * n / len(sub)})
        linhas.append({"grupo": g, "classe": "TOTAL", "n": len(sub), "pct": 100.0})
        linhas.append({"grupo": g, "classe": "encontrado no SIM (qualquer classe)", "n": int(sub.achou.sum()),
                       "pct": 100 * sub.achou.mean()})
        linhas.append({"grupo": g, "classe": "causa básica C64/C65 entre os encontrados", "n": int(sub.causa_c64.sum()),
                       "pct": 100 * sub.causa_c64.sum() / max(sub.achou.sum(), 1)})
        linhas.append({"grupo": g, "classe": "CID do paciente aparece na DO (substring), entre os encontrados",
                       "n": int(sub.cid_contido.sum()), "pct": 100 * sub.cid_contido.sum() / max(sub.achou.sum(), 1)})
        linhas.append({"grupo": g, "classe": "entrou na coorte pela rota de município do link RHC",
                       "n": int(sub.rota_mun.sum()), "pct": 100 * sub.rota_mun.mean()})
    t = pd.DataFrame(linhas)
    assert t[(t.grupo == "so_rhc") & (t.classe == "TOTAL")].n.iloc[0] == 111
    return t, p


def gravar(df: pd.DataFrame, nome: str, agregado: bool = True) -> None:
    EXP.mkdir(exist_ok=True)
    df.to_csv(EXP / nome, index=False)
    if agregado:
        AGREGADOS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXP / nome, AGREGADOS / nome)
    print(f"✅ {EXP / nome}")


def main() -> None:
    d = carregar()
    cr = captura_recaptura(d)
    dec, pac = decomposicao(d)
    aj = ajuste_falsos(dec)
    # captura-recaptura sem os prováveis falsos da lista do RHC (vínculos falsos inflam N̂)
    t = cr.iloc[0]
    b_aj = t.so_rhc - aj["so_rhc_falsos_estimados"]
    linha = {"eixo": "total", "estrato": "total, sem os prováveis falsos do RHC",
             **chapman(int(t.ambos), int(round(b_aj)), int(t.so_pipeline)), **aj}
    cr = pd.concat([cr, pd.DataFrame([linha])], ignore_index=True)
    print(cr.round(2).to_string(index=False))
    gravar(cr, "captura_recaptura_obitos.csv")
    print(dec.round(1).to_string(index=False))
    gravar(dec, "decomposicao_so_rhc.csv")
    gravar(pac, "decomposicao_so_rhc_pacientes.csv", agregado=False)  # nível de paciente: fica local
    cen = cenarios(d, cr, pac, aj)
    print(cen.round(2).to_string(index=False))
    gravar(cen, "captura_recaptura_cenarios.csv")


if __name__ == "__main__":
    main()
