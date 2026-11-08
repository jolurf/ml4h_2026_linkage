#!/usr/bin/env python3
"""Propagação do erro de linkage MEDIDO para a estimativa clínica.

A FNR não se aplica a todos os 474 óbitos: parte deles nenhum linkage produziu
(SIA com CNS, ou o `DATAOBITO` do próprio RHC, que ancora a coorte). Um vínculo
perdido só pode apagar um óbito que dependia dele. Os conjuntos, dentro do horizonte de 5 anos:

- **S_SIM** — óbito da Camada 2 (linkage demográfico ao SIM) que o RHC não
  confirma. Perder TODOS = Cenário 1b de `cenarios_degradacao_linkage.py`.
- **S_SIH** — óbito da Camada 1 cuja marca vem só de uma AIH ligada pela cascata
  R1–R4, sem óbito no SIA e sem confirmação do RHC (`concordancia_rhc_sus.py`).
- Os demais óbitos (SIA com CNS, ou confirmados pelo RHC) não dependem de
  nenhum vínculo demográfico e nunca são tocados.

Experimentos (sementes fixas):
1. **Taxa medida**: cada óbito de S_SIM ∪ S_SIH é perdido com a FNR nacional do
   estrato de completude do paciente (`metricas_por_estrato_ate2025.csv`);
   variante por região. 1 000 repetições, com OS e HR.
2. **Varredura uniforme** da perda, 0–100% em passos de 5 pontos, sobre
   S_SIM ∪ S_SIH (e só S_SIM). O ponto 100% é determinístico e é checado.
3. **Vínculos falsos**: óbitos atribuídos a sobreviventes, em número
   Binomial(|C2 em 5 anos|, r); o tempo do óbito falso é sorteado da distribuição
   dos óbitos C2 e limitado ao seguimento do próprio paciente. r = 5,7% (fan-in
   medido: 18 dos 317 vínculos de óbito disputados por mais de um paciente —
   `cardinalidade_linkage.csv`) e r = 21% (todo óbito C2 de chave ambígua, 66/314,
   tratado como falso: limite superior).
4. **Perda diferencial por raça**: FNR Preta/Parda = 1,377 × FNR Branca (razão
   nacional, `paridade_fnr_ate2025.csv`), varrendo a taxa branca.

Saídas em `experimento/` (copiadas para outputs/):
    propagacao_conjuntos.csv, propagacao_taxas_medidas.csv,
    propagacao_varredura.csv, propagacao_gap_racial.csv,
    fig_propagacao.{png,pdf}
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import gerar_relatorio_wilms as G  # noqa: E402
from cenarios_degradacao_linkage import _hr_estadio_iv  # noqa: E402
from monte_carlo_sensibilidade_linkage import _os_5a_ponto, _fnr_por_escore_nacional, _fnr_por_regiao_nacional  # noqa: E402
from nefroblastoma_ipw_cenario_b import _completude_score  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
EXP = AQUI / "experimento"
AGREGADOS = RAIZ / "outputs"
STATS = RAIZ / "dados" / "stats" / "completude_fnr_parity"
SEMENTE = 20261008
R_FALSO = {"fan-in medido (18/317)": 18 / 317, "todo ambíguo C2 (66/314)": 66 / 314}
RAZAO_FNR_RACA = 0.02459348391804996 / 0.01786303789544989


def carregar() -> pd.DataFrame:
    d, _ = G.carregar_coorte(colapsar=True)
    d = d.sort_values("id_paciente").reset_index(drop=True)
    con = duckdb.connect()
    con.register("d", d[["id_paciente"]])
    extra = con.execute(f"""
        WITH f AS (SELECT id_unico,
                SUM((subbase = 'SIH' AND obito = '1')::INT) AS aih_morte,
                SUM((subbase <> 'SIH' AND obito = '1')::INT) AS sia_obito
            FROM '{AQUI}/df_final_scored.parquet' GROUP BY id_unico)
        SELECT d.id_paciente, s.link_obito_status, s.link_obito_chave_ambigua,
               COALESCE(f.aih_morte, 0) AS aih_morte, COALESCE(f.sia_obito, 0) AS sia_obito
        FROM d LEFT JOIN '{AQUI}/obitos_scored.parquet' s ON s.id_unico = d.id_paciente
               LEFT JOIN f ON f.id_unico = d.id_paciente""").df()
    d = d.merge(extra, on="id_paciente", how="left", validate="one_to_one")
    rhc = d["evento_obito_inca"] == 1
    ev = d["E_final"] == 1
    d["S_SIM"] = ev & (d["link_obito_status"] == "obito_SIM_C2") & ~rhc
    d["S_SIH"] = ev & (d["link_obito_status"] == "obito_interno_C1") & (d.aih_morte > 0) & (d.sia_obito == 0) & ~rhc
    d["C2_5a"] = ev & (d["link_obito_status"] == "obito_SIM_C2")
    d["completude_score"] = _completude_score(d)
    return d


def medir(d: pd.DataFrame, E: np.ndarray, T: np.ndarray | None = None, com_hr: bool = False) -> dict:
    T = d["T_meses"].to_numpy() if T is None else T
    r = {"obitos": int(E.sum()), "os5": _os_5a_ponto(pd.Series(T), pd.Series(E))}
    for raca, rot in [("White", "os5_white"), ("Black or mixed", "os5_black")]:
        m = (d["Raca_Label"] == raca).to_numpy()
        r[rot] = _os_5a_ponto(pd.Series(T[m]), pd.Series(E[m]))
    r["gap"] = r["os5_white"] - r["os5_black"]
    if com_hr:
        cen = d[["cat_estadiamento"]].copy()
        cen["T"], cen["E"] = T, E
        with threadpool_limits(1):
            r["hr"] = _hr_estadio_iv(cen, "T", "E")[0]
    return r


def perder(d, p_por_paciente: np.ndarray, alvo: np.ndarray, rng) -> np.ndarray:
    E = d["E_final"].to_numpy().astype(int).copy()
    idx = np.where(alvo)[0]
    E[idx[rng.random(len(idx)) < p_por_paciente[idx]]] = 0
    return E


def acrescentar_obitos(d, E: np.ndarray, k: int, tempos: np.ndarray, rng,
                       elegiveis: np.ndarray | None = None,
                       T: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Marca óbito em k sobreviventes (entre os `elegiveis`, se dado); o tempo do
    óbito é sorteado de `tempos` e limitado ao seguimento do próprio paciente."""
    T = (d["T_meses"].to_numpy().astype(float) if T is None else T).copy()
    vivos = np.where((E == 0) & (True if elegiveis is None else elegiveis))[0]
    alvo = rng.choice(vivos, size=min(int(k), len(vivos)), replace=False)
    E = E.copy()
    E[alvo] = 1
    T[alvo] = np.minimum(T[alvo], rng.choice(tempos, size=len(alvo)))
    return E, T


def falsos(d, E: np.ndarray, r: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Atribui óbitos falsos a sobreviventes: Binomial(|C2|, r), tempo sorteado
    dos tempos de óbito C2 e limitado ao seguimento do próprio paciente."""
    k = rng.binomial(int(d["C2_5a"].sum()), r)
    return acrescentar_obitos(d, E, k, d.loc[d["C2_5a"], "T_meses"].to_numpy(), rng)


def resumo(rows: list[dict], **rotulo) -> dict:
    df = pd.DataFrame(rows)
    out = dict(rotulo, n_rep=len(df))
    for c in [c for c in ("obitos", "os5", "gap", "hr") if c in df]:
        out[f"{c}_med"] = df[c].median()
        out[f"{c}_p025"] = df[c].quantile(.025)
        out[f"{c}_p975"] = df[c].quantile(.975)
    return out


def main() -> None:
    d = carregar()
    alvo = (d["S_SIM"] | d["S_SIH"]).to_numpy()
    ref = medir(d, d["E_final"].to_numpy(), com_hr=True)
    assert ref["obitos"] == 474 and abs(ref["os5"] - 80.49) < 0.05, ref
    E1b = np.where(d["S_SIM"], 0, d["E_final"]).astype(int)
    r1b = medir(d, E1b, com_hr=True)
    assert r1b["obitos"] == 416, r1b  # = Cenário 1b de cenarios_degradacao_linkage.py

    conj = pd.DataFrame([
        {"conjunto": "óbitos em 5 anos (referência)", "n": int(d["E_final"].sum())},
        {"conjunto": "S_SIM: C2 sem confirmação do RHC", "n": int(d["S_SIM"].sum())},
        {"conjunto": "S_SIH: C1 só de AIH, sem confirmação do RHC", "n": int(d["S_SIH"].sum())},
        {"conjunto": "dependentes de linkage demográfico (S_SIM ∪ S_SIH)", "n": int(alvo.sum())},
        {"conjunto": "óbitos C2 em 5 anos (base dos falsos vínculos)", "n": int(d["C2_5a"].sum())},
    ])
    print(conj.to_string(index=False))

    m = pd.read_csv(STATS / "metricas_por_estrato_ate2025.csv")
    fnr_esc = _fnr_por_escore_nacional(m)
    fnr_reg = _fnr_por_regiao_nacional(m)
    p_esc = d["completude_score"].map(fnr_esc).fillna(fnr_esc[3]).to_numpy()
    p_reg = d["regiao_residencia"].map(fnr_reg).fillna(m.set_index("estrato").loc["Nacional", "fnr"]).to_numpy()

    rng = np.random.default_rng(SEMENTE)
    taxas = [dict(cenario="0 referência", n_rep=1, obitos_med=ref["obitos"], os5_med=ref["os5"],
                  gap_med=ref["gap"], hr_med=ref["hr"]),
             dict(cenario="1b sem link SIM (100% de S_SIM perdido)", n_rep=1, obitos_med=r1b["obitos"],
                  os5_med=r1b["os5"], gap_med=r1b["gap"], hr_med=r1b["hr"])]
    for nome, p in [("FNR medida por completude", p_esc), ("FNR medida por região", p_reg)]:
        taxas.append(resumo([medir(d, perder(d, p, alvo, rng), com_hr=True) for _ in range(1000)],
                            cenario=nome))
        for rot, r in R_FALSO.items():
            rows = []
            for _ in range(1000):
                E = perder(d, p, alvo, rng)
                E, T = falsos(d, E, r, rng)
                rows.append(medir(d, E, T, com_hr=True))
            taxas.append(resumo(rows, cenario=f"{nome} + vínculos falsos, {rot}"))
    # Perdas uniformes de outras fontes de medida (gerador próprio, para não mexer nos
    # sorteios das linhas acima nem da varredura abaixo, já citadas):
    #   - a perda da cascata R1→R3→R4 com CNS escondido (1 − recall, pool de coorte);
    #   - a perda do pipeline implicada pela captura-recaptura (1 − captura do pipeline).
    # A FNR do BI×AQ é perda DENTRO do SIA: é piso, não estimativa, da perda do passo SIM.
    rng2 = np.random.default_rng(SEMENTE + 1)
    ca = pd.read_csv(AGREGADOS / "cascata_registro_retido_ate2025.csv")
    casc = ca[(ca.uf == "todas") & (ca.pool == "1% dos pacientes (tamanho de coorte)")
              & (ca.ordem == "produção (R1→R3→R4)") & (ca.regra == "cascata (acumulada)")].iloc[0]
    cr_csv = EXP / "captura_recaptura_obitos.csv"
    perdas = {"perda da cascata com CNS escondido": 1 - casc.recall}
    if cr_csv.exists():
        cr = pd.read_csv(cr_csv)
        perdas["perda do pipeline implicada pela captura-recaptura"] = 1 - cr.iloc[0].captura_pipeline
    for nome, p_unif in perdas.items():
        r = resumo([medir(d, perder(d, np.full(len(d), p_unif), alvo, rng2), com_hr=True) for _ in range(1000)],
                   cenario=nome)
        r["taxa_uniforme"] = p_unif
        taxas.append(r)
    tx = pd.DataFrame(taxas)
    print(tx.round(2).to_string(index=False))

    varr = []
    for nome_alvo, a in [("S_SIM ∪ S_SIH", alvo), ("S_SIM", d["S_SIM"].to_numpy())]:
        for p in np.round(np.arange(0, 1.0001, 0.05), 2):
            rows = [medir(d, perder(d, np.full(len(d), p), a, rng)) for _ in range(1 if p in (0, 1) else 200)]
            varr.append(resumo(rows, alvo=nome_alvo, p_perda=p))
    for r in np.round(np.arange(0, 0.2501, 0.025), 3):
        rows = [medir(d, *falsos(d, d["E_final"].to_numpy().astype(int), r, rng)) for _ in range(200)]
        varr.append(resumo(rows, alvo="vínculos falsos (sobreviventes)", p_perda=r))
    vr = pd.DataFrame(varr)
    fim = vr[(vr.alvo == "S_SIM") & (vr.p_perda == 1)].iloc[0]
    assert abs(fim["os5_med"] - r1b["os5"]) < 1e-9, "100% de perda em S_SIM tem de ser o Cenário 1b"

    gap = []
    branco = (d["Raca_Label"] == "White").to_numpy()
    for pw in np.round(np.arange(0, 0.3001, 0.025), 3):
        p = np.where(branco, pw, min(1.0, pw * RAZAO_FNR_RACA))
        gap.append(resumo([medir(d, perder(d, p, alvo, rng)) for _ in range(200)],
                          fnr_branca=pw, fnr_preta_parda=min(1.0, pw * RAZAO_FNR_RACA)))
    gp = pd.DataFrame(gap)

    def limiar(df: pd.DataFrame, delta: float) -> float:
        s = df.sort_values("p_perda")
        x, y = s["p_perda"].to_numpy(), (s["os5_med"] - ref["os5"]).to_numpy()
        return float(np.interp(delta, y, x)) if y.max() >= delta else np.nan
    vu = vr[vr.alvo == "S_SIM ∪ S_SIH"]
    lim = pd.DataFrame([{"deslocamento_os_pp": dl, "perda_necessaria": limiar(vu, dl),
                         "deslocamento_maximo_possivel_pp": vu["os5_med"].max() - ref["os5"]}
                        for dl in (0.5, 1.0, 2.0, 4.0)])
    print(lim.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for nome, cor in [("S_SIM ∪ S_SIH", "#1f4e79"), ("S_SIM", "#7aa6c2")]:
        s = vr[vr.alvo == nome]
        ax[0].plot(100 * s.p_perda, s.os5_med, color=cor, label=f"loss of {nome.replace('∪', 'or')} death links")
        ax[0].fill_between(100 * s.p_perda, s.os5_p025, s.os5_p975, color=cor, alpha=.2)
    for y, rot, ls in [(ref["os5"], "Scenario 0", "--"), (89.1, "Scenario 1 (claims only)", ":")]:
        ax[0].axhline(y, color="0.3", ls=ls, lw=1)
        ax[0].text(1, y + .2, rot, fontsize=9, color="0.3")
    ax[0].set_xlabel("Share of linkage-dependent death links lost (%)", fontsize=10)
    ax[0].set_ylabel("5-year overall survival (%)", fontsize=10)
    ax[0].legend(fontsize=9, loc="center right")
    ax[1].plot(100 * gp.fnr_branca, gp.gap_med, color="#1f4e79")
    ax[1].fill_between(100 * gp.fnr_branca, gp.gap_p025, gp.gap_p975, color="#1f4e79", alpha=.2)
    ax[1].axhline(ref["gap"], color="0.3", ls="--", lw=1)
    ax[1].set_xlabel("FNR, White patients (%); Black/mixed = 1.38×", fontsize=10)
    ax[1].set_ylabel("White − Black/mixed 5-year OS gap (pp)", fontsize=10)
    for a in ax:
        a.tick_params(labelsize=9)
    fig.tight_layout()
    EXP.mkdir(exist_ok=True)
    fig.savefig(EXP / "fig_propagacao.png", dpi=200)
    fig.savefig(EXP / "fig_propagacao.pdf")

    AGREGADOS.mkdir(parents=True, exist_ok=True)
    for df, nome in [(conj, "propagacao_conjuntos.csv"), (tx, "propagacao_taxas_medidas.csv"),
                     (vr, "propagacao_varredura.csv"), (gp, "propagacao_gap_racial.csv"),
                     (lim, "propagacao_limiares.csv")]:
        df.to_csv(EXP / nome, index=False)
        shutil.copy2(EXP / nome, AGREGADOS / nome)
        print(f"✅ {EXP / nome}")


if __name__ == "__main__":
    sys.exit(main())
