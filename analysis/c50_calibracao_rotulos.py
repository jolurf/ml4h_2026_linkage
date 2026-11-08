"""Um modelo treinado sob três fontes de óbito: o que muda no modelo e na
calibração.

Modelo: o Cox multivariado do C50 (`src/c50/analises.cox`) — mesmas
covariáveis (`config.COVARIAVEIS_COX` + `ivs_z`), mesma categorização, sem
penalização, ajustado dentro de `ajuste_deterministico()`. Interpretável por
construção: cada coeficiente é um HR publicado lado a lado nos três rótulos. A
matriz de desenho é montada aqui (e não com `cox()`, que não a devolve e é
travada por teste) para que a calibração possa ser avaliada.

População: pacientes da coorte de sobrevida do C50 ligados ao RHC (223 463), a
única em que as três fontes existem; só estádios invasivos I–IV (em `cox()` o
estádio 0 vira NaN depois do `dropna` e cai no grupo de referência — aqui é
excluído explicitamente). Tempos em meses por `date_diff('month', ...)`, a mesma
conta do `coorte.montar`, com fim administrativo em 31/12/2024 para os três.

Rótulos de óbito:
    SIM   — `obito`/`data_morte` da coorte (linkage demográfico ao SIM);
    RHC   — `dt_obito_rhc` do registro hospitalar de câncer;
    União — o primeiro dos dois.

Desenho: treino/teste 70/30 estratificado por estádio, semente fixa. Cada modelo
(treinado num rótulo) é avaliado no teste contra cada rótulo: C de Harrell e, aos
60 meses, calibração — risco previsto médio contra risco observado por
Kaplan–Meier (calibração na média) e o ICI de Austin et al. 2020
(`lifelines.calibration.survival_probability_calibration`), no total e por raça
e região.

Saídas em projetos/C50/resultados/ (+ outputs/):
    calibracao_rotulos_hr.csv, calibracao_rotulos_global.csv,
    calibracao_rotulos_subgrupos.csv, calibracao_rotulos_populacao.csv
"""
from __future__ import annotations

import shutil

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.calibration import survival_probability_calibration  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402

import _raiz  # noqa: F401,E402
import paths as _paths  # noqa: E402
from c50 import config  # noqa: E402
from c50.analises import _colapsar_rhc_por_paciente, ajuste_deterministico  # noqa: E402

BASE = _paths.PROJETOS / "C50"
OUT = BASE / "resultados"
AGREGADOS = _paths.PROJETOS.parent / "outputs"
FIM = "2024-12-31"
T0 = 60
SEMENTE = 20261008
COVS = list(config.COVARIAVEIS_COX) + ["ivs_z"]
ROTULOS = ("SIM", "RHC", "Uniao")


def carregar() -> pd.DataFrame:
    vinc = pd.read_csv(BASE / "resultados" / "tabela_vinculo_rhc_sus.csv", sep=";", decimal=",",
                       encoding="utf-8-sig", dtype={"id_unico": str})
    vinc["dt_obito_rhc"] = pd.to_datetime(vinc["dt_obito_rhc"], errors="coerce")
    rhc = _colapsar_rhc_por_paciente(vinc)
    con = duckdb.connect()
    con.register("rhc", rhc[["id_unico", "obito_rhc", "dt_obito_rhc"]])
    d = con.execute(f"""
        SELECT c.id_unico, {', '.join(COVS)}, c.data_diagnostico, c.obito, c.data_morte, r.obito_rhc,
               r.dt_obito_rhc,
               -- SIM
               (c.obito = 1 AND c.data_morte <= DATE '{FIM}')::INT AS E_SIM,
               date_diff('month', c.data_diagnostico,
                   LEAST(CASE WHEN c.obito = 1 THEN c.data_morte ELSE DATE '{FIM}' END, DATE '{FIM}')) AS T_SIM,
               -- RHC
               (r.obito_rhc = 1 AND r.dt_obito_rhc <= DATE '{FIM}')::INT AS E_RHC,
               date_diff('month', c.data_diagnostico,
                   LEAST(COALESCE(r.dt_obito_rhc, DATE '{FIM}'), DATE '{FIM}')) AS T_RHC
        FROM '{BASE}/intermediarios/coorte_sobrevida.parquet' c JOIN rhc r USING (id_unico)
        ORDER BY c.id_unico""").df()
    n_ligados = len(d)
    # União: o primeiro óbito entre as duas fontes
    t_sim = np.where(d.E_SIM == 1, d.T_SIM, np.inf)
    t_rhc = np.where(d.E_RHC == 1, d.T_RHC, np.inf)
    d["E_Uniao"] = ((d.E_SIM == 1) | (d.E_RHC == 1)).astype(int)
    d["T_Uniao"] = np.where(d.E_Uniao == 1, np.minimum(t_sim, t_rhc), np.maximum(d.T_SIM, d.T_RHC))
    pop = [("ligados RHC × coorte de sobrevida", n_ligados)]
    d = d[d.staging.isin(config.ESTADIOS_INVASIVOS)]
    pop.append(("estádio invasivo I–IV", len(d)))
    d = d[(d[[f"T_{r}" for r in ROTULOS]] >= 0).all(axis=1)]
    pop.append(("tempo ≥ 0 nas três fontes (exclui óbito RHC antes do diagnóstico SUS)", len(d)))
    d = d.dropna(subset=COVS)
    pop.append(("covariáveis completas", len(d)))
    d.attrs["populacao"] = pop
    return d.reset_index(drop=True)


def desenho(d: pd.DataFrame) -> pd.DataFrame:
    X = d[COVS].copy()
    X["staging"] = pd.Categorical(X["staging"], categories=config.ESTADIOS_INVASIVOS)
    racas = [config.RACA_REFERENCIA] + sorted(r for r in X.raca_label.unique() if r != config.RACA_REFERENCIA)
    X["raca_label"] = pd.Categorical(X["raca_label"], categories=racas)
    X["regiao"] = pd.Categorical(X["regiao"], categories=sorted(X.regiao.unique()))
    return pd.get_dummies(X, columns=["staging", "raca_label", "regiao"], drop_first=True, dtype=float)


def calibracao_decis(risco: np.ndarray, T: np.ndarray, E: np.ndarray, k: int = 10) -> tuple[float, float]:
    """Calibração por decis de risco previsto: |previsto médio − observado KM| por decil.
    Devolve (erro absoluto médio, erro absoluto máximo), em pontos percentuais.
    Interpretável e estável; o ICI do spline do lifelines falhou num dos nove ajustes."""
    ordem = np.argsort(risco, kind="stable")
    erros = []
    for parte in np.array_split(ordem, k):
        erros.append(abs(risco[parte].mean() - risco_km(T[parte], E[parte])))
    return 100 * float(np.mean(erros)), 100 * float(np.max(erros))


def risco_km(T, E, t0=T0) -> float:
    return 1 - float(KaplanMeierFitter().fit(T, E).survival_function_at_times(t0).iloc[0])


def main() -> None:
    d = carregar()
    X = desenho(d)
    rng = np.random.default_rng(SEMENTE)
    teste = np.zeros(len(d), bool)
    for _, idx in d.groupby("staging").indices.items():
        teste[rng.choice(idx, size=int(round(0.3 * len(idx))), replace=False)] = True
    pop = pd.DataFrame(d.attrs["populacao"], columns=["passo", "n"])
    pop.loc[len(pop)] = ["treino (70%)", int((~teste).sum())]
    pop.loc[len(pop)] = ["teste (30%)", int(teste.sum())]
    for r in ROTULOS:
        pop.loc[len(pop)] = [f"óbitos ({r}), população analisada", int(d[f"E_{r}"].sum())]
    print(pop.to_string(index=False))

    modelos, hr = {}, []
    for r in ROTULOS:
        tr = X[~teste].assign(T=d.loc[~teste, f"T_{r}"].values, E=d.loc[~teste, f"E_{r}"].values)
        m = CoxPHFitter(penalizer=config.COX_PENALIZER)
        with ajuste_deterministico():
            m.fit(tr, duration_col="T", event_col="E")
        modelos[r] = m
        s = m.summary
        hr.append(pd.DataFrame({"rotulo_treino": r, "covariavel": s.index, "HR": s["exp(coef)"].values,
                                "ic95_inf": s["exp(coef) lower 95%"].values,
                                "ic95_sup": s["exp(coef) upper 95%"].values}))
    hr = pd.concat(hr).pivot(index="covariavel", columns="rotulo_treino", values=["HR", "ic95_inf", "ic95_sup"])
    hr.columns = [f"{a}_{b}" for a, b in hr.columns]
    hr = hr.reset_index()
    print(hr[["covariavel"] + [f"HR_{r}" for r in ROTULOS]].round(3).to_string(index=False))

    te = d[teste].reset_index(drop=True)
    Xte = X[teste].reset_index(drop=True)
    glob, sub = [], []
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey=True)
    for i, rt in enumerate(ROTULOS):
        m = modelos[rt]
        risco = 1 - m.predict_survival_function(Xte, times=[T0]).T.iloc[:, 0].to_numpy()
        ph = m.predict_partial_hazard(Xte).to_numpy()
        for j, ra in enumerate(ROTULOS):
            T, E = te[f"T_{ra}"].to_numpy(), te[f"E_{ra}"].to_numpy()
            obs = risco_km(T, E)
            # a curva de calibração (spline de Austin et al.) exige T > 0: óbito no mês
            # do diagnóstico (T = 0 em date_diff mensal) entra com meio mês
            ava = Xte.assign(T=np.maximum(T, 0.5), E=E)
            with ajuste_deterministico():
                _, ici, e50 = survival_probability_calibration(m, ava, t0=T0, ax=axes[i, j])
            axes[i, j].set_title(f"train {rt} → eval {ra}", fontsize=9)
            axes[i, j].legend().remove() if axes[i, j].get_legend() else None
            dec_mae, dec_max = calibracao_decis(risco, T, E)
            glob.append({"rotulo_treino": rt, "rotulo_avaliacao": ra, "n_teste": len(te),
                         "cal_decil_mae_pp": dec_mae, "cal_decil_max_pp": dec_max,
                         "obitos_teste": int(E.sum()), "c_harrell": concordance_index(T, -ph, E),
                         "risco_5a_previsto_medio": 100 * risco.mean(), "risco_5a_observado_km": 100 * obs,
                         "calibracao_na_media_pp": 100 * (risco.mean() - obs),
                         "razao_previsto_observado": risco.mean() / obs, "ici": ici, "e50": e50})
            for eixo in ("raca_label", "regiao"):
                for g, idx in te.groupby(eixo).indices.items():
                    if len(idx) < 500:
                        continue
                    o = risco_km(T[idx], E[idx])
                    sub.append({"rotulo_treino": rt, "rotulo_avaliacao": ra, "eixo": eixo, "grupo": g,
                                "n": len(idx), "obitos": int(E[idx].sum()),
                                "risco_5a_previsto_medio": 100 * risco[idx].mean(),
                                "risco_5a_observado_km": 100 * o,
                                "calibracao_na_media_pp": 100 * (risco[idx].mean() - o),
                                "razao_previsto_observado": risco[idx].mean() / o})
    for a in axes[-1]:
        a.set_xlabel("Predicted 5-year risk", fontsize=9)
    for a in axes[:, 0]:
        a.set_ylabel("Observed 5-year risk", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibracao_rotulos.png", dpi=180)
    fig.savefig(OUT / "fig_calibracao_rotulos.pdf")
    glob, sub = pd.DataFrame(glob), pd.DataFrame(sub)
    print(glob.round(3).to_string(index=False))
    pior = (sub.assign(abs_cal=sub.calibracao_na_media_pp.abs())
            .sort_values("abs_cal", ascending=False).groupby(["rotulo_treino", "rotulo_avaliacao"]).head(1))
    print(pior.round(2).to_string(index=False))

    AGREGADOS.mkdir(parents=True, exist_ok=True)
    for df, nome in [(hr, "calibracao_rotulos_hr.csv"), (glob, "calibracao_rotulos_global.csv"),
                     (sub, "calibracao_rotulos_subgrupos.csv"), (pop, "calibracao_rotulos_populacao.csv")]:
        df.to_csv(OUT / nome, index=False)
        shutil.copy2(OUT / nome, AGREGADOS / nome)
        print(f"✅ {OUT / nome}")


if __name__ == "__main__":
    main()
