#!/usr/bin/env python3
"""Tabela 2: gap racial na sobrevida com raça faltante.

Um IPW cujos pesos fossem função do escore de completude cadastral não serve
aqui: nesta coorte, restrita a quem tem raça informada, o escore só se afasta do
valor modal em 15 de 2 503 pacientes, e o peso desses 15 é ~0,007 — IPW e CC
sairiam iguais (HR 0,7399 e 0,7399). E esse peso modela a linkabilidade, não a
variável que de fato falta na Tabela 2, que é a raça.

Aqui a variável faltante é a raça (119 pacientes sem registro), e os três
métodos tratam dela:

- **CC** — só raça observada (n = 2 503).
- **IPW** — peso 1/P̂(raça observada | X), estabilizado pela proporção MARGINAL
  de observados, truncado em p1/p99; Cox ponderado com variância robusta
  (sanduíche); KM ponderado para a OS.
- **MI** — 20 imputações próprias: logística Branca × Preta/Parda nos
  observados, β sorteado de N(β̂, V̂), raça sorteada de Bernoulli(p̂); o
  desfecho entra no modelo de imputação (E aos 5 anos e o Nelson–Aalen H(T),
  White & Royston 2009). Regras de Rubin no log HR e no gap de OS.

X = idade, sexo, estádio (com `Ignored` como categoria), região de residência
(com `Ignorado`), ano do diagnóstico e grupo de tratamento. A população dos
três métodos é a mesma: Branca + Preta/Parda + raça faltante (n = 2 622);
Amarela e Indígena ficam fora.

O estimando: HR univariado Branca vs Preta/Parda e
diferença de OS em 5 anos (pontos percentuais).

Saídas em `experimento/` (copiadas para outputs/):
    tabela2_revisada.csv, tabela2_pesos_ipw.csv
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from scipy import stats
from threadpoolctl import threadpool_limits

import gerar_relatorio_wilms as G

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
EXP = AQUI / "experimento"
AGREGADOS = AQUI.parents[1] / "outputs"
TAU = G.TAU
N_IMP = 20
SEMENTE = 20261008


def preparar() -> pd.DataFrame:
    d, _ = G.carregar_coorte(colapsar=True)
    d = d[d["Raca_Label"].isin(["White", "Black or mixed"]) | d["Raca_Label"].isna()].copy()
    d = d.sort_values("id_paciente").reset_index(drop=True)
    d["observado"] = d["Raca_Label"].notna().astype(int)
    d["white"] = np.where(d["observado"] == 1, (d["Raca_Label"] == "White").astype(float), np.nan)
    d["regiao"] = d["regiao_residencia"].fillna("Ignorado")
    trat = d["Tratamento_Grupo"].where(d["Tratamento_Grupo"].map(d["Tratamento_Grupo"].value_counts()) >= 30,
                                       "Outros")
    d["trat"] = trat
    naf = NelsonAalenFitter().fit(d["T_meses"], d["E_final"])
    d["H"] = naf.cumulative_hazard_at_times(d["T_meses"]).values
    return d


def matriz(d: pd.DataFrame, com_desfecho: bool) -> pd.DataFrame:
    X = pd.get_dummies(d[["cat_sexo", "cat_estadiamento", "regiao", "trat"]], drop_first=True, dtype=float)
    X["idade"] = d["IDADE_INT"].astype(float)
    X["ano"] = d["ANO_DIAGN"].astype(float) - 2015
    if com_desfecho:
        X["E"] = d["E_final"].astype(float)
        X["H"] = d["H"].astype(float)
    return sm.add_constant(X, has_constant="add")


def hr_e_gap(T, E, white, w=None) -> dict:
    df = pd.DataFrame({"T": T, "E": E, "white": white})
    if w is not None:
        df["w"] = w
    with threadpool_limits(1):
        cph = CoxPHFitter().fit(df, "T", "E", weights_col="w" if w is not None else None,
                                robust=w is not None)
    s = cph.summary.loc["white"]
    os5 = {}
    for g in (1, 0):
        m = df["white"] == g
        kmf = KaplanMeierFitter().fit(df.loc[m, "T"], df.loc[m, "E"],
                                      weights=df.loc[m, "w"] if w is not None else None)
        os5[g] = float(kmf.survival_function_at_times(TAU).iloc[0])
    return {"loghr": s["coef"], "se": s["se(coef)"], "os_white": os5[1], "os_black": os5[0],
            "gap": 100 * (os5[1] - os5[0])}


def var_gap_km(T, E, white) -> float:
    """Var do gap de OS (pp²) por Greenwood, para as regras de Rubin."""
    v = 0.0
    for g in (1, 0):
        m = white == g
        kmf = KaplanMeierFitter().fit(T[m], E[m])
        ci = kmf.confidence_interval_survival_function_
        idx = ci.index.asof(TAU)
        v += ((ci.loc[idx].iloc[1] - ci.loc[idx].iloc[0]) / (2 * 1.96)) ** 2
    return v * 1e4


def linha(metodo, n, r, lo_hi_gap, nota, n_imp=None) -> dict:
    return {"metodo": metodo, "n": n, "n_imputacoes": n_imp,
            "hr_white_vs_black": np.exp(r["loghr"]),
            "hr_ic_lo": np.exp(r["loghr"] - 1.96 * r["se"]) if "lo" not in r else np.exp(r["lo"]),
            "hr_ic_hi": np.exp(r["loghr"] + 1.96 * r["se"]) if "hi" not in r else np.exp(r["hi"]),
            "os5_white": 100 * r["os_white"], "os5_black": 100 * r["os_black"],
            "gap_pp": r["gap"], "gap_ic_lo": lo_hi_gap[0], "gap_ic_hi": lo_hi_gap[1], "nota": nota}


def main() -> None:
    d = preparar()
    obs = d[d["observado"] == 1]
    print(f"população: {len(d)} | raça observada: {len(obs)} | faltante: {int((d.observado == 0).sum())}")
    linhas = []

    # CC
    r = hr_e_gap(obs["T_meses"], obs["E_final"], obs["white"])
    vg = var_gap_km(obs["T_meses"].values, obs["E_final"].values, obs["white"].values)
    linhas.append(linha("Complete case", len(obs), r, (r["gap"] - 1.96 * vg**.5, r["gap"] + 1.96 * vg**.5),
                        "raça observada"))

    # IPW para P(raça observada)
    Xr = matriz(d, com_desfecho=True)
    pr = sm.GLM(d["observado"], Xr, family=sm.families.Binomial()).fit().predict(Xr)
    w = d["observado"].mean() / pr
    lo, hi = np.percentile(w[d.observado == 1], [1, 99])
    w = w.clip(lo, hi)
    wo = w[d.observado == 1]
    r = hr_e_gap(obs["T_meses"], obs["E_final"], obs["white"], wo.values)
    # IC do gap por bootstrap de pacientes refazendo o modelo de pesos
    rng = np.random.default_rng(SEMENTE)
    gaps = []
    for _ in range(500):
        b = d.sample(len(d), replace=True, random_state=rng.integers(1 << 31)).reset_index(drop=True)
        Xb = matriz(b, com_desfecho=True).reindex(columns=Xr.columns, fill_value=0.0)
        try:
            pb = sm.GLM(b["observado"], Xb, family=sm.families.Binomial()).fit().predict(Xb)
        except Exception:
            continue
        wb = (b["observado"].mean() / pb).clip(lo, hi)
        ob = b["observado"] == 1
        os5 = {}
        for g in (1, 0):
            m = ob & (b["white"] == g)
            os5[g] = KaplanMeierFitter().fit(b.loc[m, "T_meses"], b.loc[m, "E_final"],
                                             weights=wb[m]).survival_function_at_times(TAU).iloc[0]
        gaps.append(100 * (os5[1] - os5[0]))
    linhas.append(linha("IPW, P(raça observada)", len(obs), r, tuple(np.percentile(gaps, [2.5, 97.5])),
                        "Cox ponderado, variância robusta; IC do gap por bootstrap (500)"))
    pesos = pd.DataFrame([{"n_peso_diferente_de_1": int((np.abs(wo - 1) > 1e-6).sum()),
                           "min": wo.min(), "p25": wo.quantile(.25), "mediana": wo.median(),
                           "p75": wo.quantile(.75), "max": wo.max(), "trunc_p1": lo, "trunc_p99": hi,
                           "auc_aprox_modelo_observacao": np.nan}])

    # MI
    Xm = matriz(d, com_desfecho=True)
    mod = sm.GLM(obs["white"], Xm.loc[obs.index], family=sm.families.Binomial()).fit()
    rng = np.random.default_rng(SEMENTE)
    falt = d.index[d.observado == 0]
    res = []
    for _ in range(N_IMP):
        beta = rng.multivariate_normal(mod.params.values, mod.cov_params().values)
        p = 1 / (1 + np.exp(-(Xm.loc[falt].values @ beta)))
        wh = d["white"].copy()
        wh.loc[falt] = rng.binomial(1, p).astype(float)
        r_i = hr_e_gap(d["T_meses"], d["E_final"], wh)
        r_i["var_gap"] = var_gap_km(d["T_meses"].values, d["E_final"].values, wh.values)
        res.append(r_i)
    R = pd.DataFrame(res)
    m = len(R)

    def rubin(est, var):
        qbar, ubar, b = est.mean(), var.mean(), est.var(ddof=1)
        t = ubar + (1 + 1 / m) * b
        gl = (m - 1) * (1 + ubar / ((1 + 1 / m) * b)) ** 2
        q = stats.t.ppf(.975, gl)
        return qbar, qbar - q * t**.5, qbar + q * t**.5
    lhr, llo, lhi = rubin(R["loghr"], R["se"] ** 2)
    g, glo, ghi = rubin(R["gap"], R["var_gap"])
    linhas.append(linha("Imputação múltipla", len(d),
                        {"loghr": lhr, "lo": llo, "hi": lhi, "os_white": R.os_white.mean(),
                         "os_black": R.os_black.mean(), "gap": g}, (glo, ghi),
                        "logística c/ desfecho (E, H(T)); regras de Rubin", n_imp=m))

    t = pd.DataFrame(linhas)
    # a CC tem de reproduzir o HR univariado do cenário A (raça observada)
    ref = pd.read_csv(EXP / "resumo_hr_cenario_ab.csv").iloc[0]["hr_white_vs_black"]
    assert abs(t.loc[0, "hr_white_vs_black"] - ref) < 1e-6, (t.loc[0, "hr_white_vs_black"], ref)
    print(t.round(3).to_string(index=False))
    print(pesos.round(3).to_string(index=False))
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    for df, nome in [(t, "tabela2_revisada.csv"), (pesos, "tabela2_pesos_ipw.csv")]:
        df.to_csv(EXP / nome, index=False)
        shutil.copy2(EXP / nome, AGREGADOS / nome)
        print(f"✅ {EXP / nome}")


if __name__ == "__main__":
    main()
