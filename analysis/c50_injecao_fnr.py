"""O erro de linkage MEDIDO dentro do modelo treinado: as taxas de erro
medidas por grupo alimentadas no modelo de sobrevida do C50.

`c50_calibracao_rotulos.py` varia a FONTE do rótulo. Aqui a fonte é a mais
completa (União pipeline ∪ RHC) e o que varia é a perda de vínculo. Só os óbitos
que dependem do link demográfico ao SIM podem ser perdidos: camada C2 do
`vincular_sim` e vivos no RHC. Os da camada C1 (flag de óbito no próprio registro
de AQ/AR/SIH) não cruzam identidade e ficam fora — sortear também sobre eles
superestimaria a perda. Cada óbito elegível é perdido com a
taxa de falso negativo nacional da raça da paciente (`metricas_por_estrato_ate2025.csv`,
linkage cego BI×AQ; piso da perda do passo SIM), com a perda da cascata testada com
o CNS escondido (1 − recall, `cascata_registro_retido_ate2025.csv`) e com 5× a FNR
(estresse); vira censura no fim do seguimento administrativo. O Cox é reajustado
em cada sorteio; reportam-se os HR e a calibração contra o rótulo completo.

Reusa `carregar`, `desenho` e `calibracao_decis` de `c50_calibracao_rotulos.py`
(mesma população, mesmo split, mesmo modelo).

Saída: projetos/C50/resultados/injecao_fnr_c50.csv (+ outputs/)
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import duckdb

import _raiz  # noqa: F401
import paths as _paths
from c50 import config
from c50.analises import ajuste_deterministico
from c50_calibracao_rotulos import BASE, OUT, AGREGADOS, SEMENTE, T0, calibracao_decis, carregar, desenho

N_SORTEIOS = 50
MAPA_RACA = {"Branca": "Branca", "Parda": "Parda", "Preta": "Preta", "Amarela": "Amarela",
             "Indígena": "Indígena", "Ignorada": "Sem informação"}


def main() -> None:
    d = carregar()
    X = desenho(d)
    rng = np.random.default_rng(SEMENTE)
    teste = np.zeros(len(d), bool)
    for _, idx in d.groupby("staging").indices.items():
        teste[rng.choice(idx, size=int(round(0.3 * len(idx))), replace=False)] = True

    m = pd.read_csv(_paths.STATS / "completude_fnr_parity" / "metricas_por_estrato_ate2025.csv")
    fnr = {e.split(": ")[1]: f for e, f in zip(m.estrato, m.fnr) if e.startswith("Raça/cor:")}
    p = d["raca_label"].map(MAPA_RACA).map(fnr).fillna(m.set_index("estrato").loc["Nacional", "fnr"]).to_numpy()
    con = duckdb.connect()
    con.register("d", d[["id_unico"]])
    camada = con.execute(f"""SELECT d.id_unico, c.camada_obito FROM d
        JOIN '{BASE}/intermediarios/coorte_sobrevida.parquet' c USING (id_unico)""").df()
    d["camada_obito"] = d["id_unico"].map(camada.set_index("id_unico")["camada_obito"])
    # óbito que só existe pelo link ao SIM: camada C2 e vivo no RHC
    so_sim = ((d.E_SIM == 1) & (d.E_RHC == 0) & (d.camada_obito == 2)).to_numpy()
    ca = pd.read_csv(AGREGADOS / "cascata_registro_retido_ate2025.csv")
    casc = ca[(ca.uf == "todas") & (ca.pool == "1% dos pacientes (tamanho de coorte)")
              & (ca.ordem == "produção (R1→R3→R4)") & (ca.regra == "cascata (acumulada)")].iloc[0]
    perda_cascata = 1 - casc.recall
    T_cens = np.maximum(d.T_SIM, d.T_RHC).to_numpy()       # vivo no RHC: seguimento até o fim administrativo
    E0, T0_ = d.E_Uniao.to_numpy(), d.T_Uniao.to_numpy().astype(float)
    print(f"população {len(d):,}; óbitos União {E0.sum():,}; só pelo link SIM (C2) {so_sim.sum():,}; "
          f"perda da cascata {perda_cascata:.3f}")

    def ajustar(E, T):
        tr = X[~teste].assign(T=T[~teste], E=E[~teste])
        mod = CoxPHFitter(penalizer=config.COX_PENALIZER)
        with ajuste_deterministico():
            mod.fit(tr, duration_col="T", event_col="E")
        risco = 1 - mod.predict_survival_function(X[teste], times=[T0]).T.iloc[:, 0].to_numpy()
        mae, _ = calibracao_decis(risco, T0_[teste], E0[teste])  # contra o rótulo COMPLETO
        s = mod.summary["exp(coef)"]
        return {"hr_estadio_iv": s["staging_IV"], "hr_parda": s["raca_label_Parda"],
                "hr_preta": s["raca_label_Preta"], "cal_decil_mae_pp_vs_completo": mae,
                "risco_5a_previsto_medio": 100 * risco.mean()}

    linhas = [{"cenario": "referência (União, sem perda)", "sorteio": 0, "obitos_perdidos": 0,
               "obitos_elegiveis": int(so_sim.sum()), **ajustar(E0, T0_)}]
    taxas = [("FNR medida por raça", p), ("perda da cascata com CNS escondido", np.full(len(d), perda_cascata)),
             ("5× a FNR medida (estresse)", np.minimum(1, 5 * p))]
    for rot, taxa in taxas:
        for s in range(N_SORTEIOS):
            perde = so_sim & (rng.random(len(d)) < taxa)
            E, T = E0.copy(), T0_.copy()
            E[perde], T[perde] = 0, T_cens[perde]
            linhas.append({"cenario": rot, "sorteio": s + 1, "obitos_perdidos": int(perde.sum()),
                           "obitos_elegiveis": int(so_sim.sum()), **ajustar(E, T)})
        print(f"  {rot}: ok", flush=True)
    t = pd.DataFrame(linhas)
    ref = t.iloc[0]
    assert abs(ref.hr_estadio_iv - 4.31) < 0.01, ref  # = HR da União em c50_calibracao_rotulos
    res = t.groupby("cenario", sort=False).agg(
        n=("sorteio", "size"), obitos_perdidos=("obitos_perdidos", "median"),
        obitos_elegiveis=("obitos_elegiveis", "first"),
        **{f"{c}_{q}": (c, f) for c in ("hr_estadio_iv", "hr_parda", "hr_preta", "cal_decil_mae_pp_vs_completo")
           for q, f in (("med", "median"), ("p025", lambda x: x.quantile(.025)), ("p975", lambda x: x.quantile(.975)))}
    ).reset_index()
    print(res.round(3).T.to_string())
    res.to_csv(OUT / "injecao_fnr_c50.csv", index=False)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "injecao_fnr_c50.csv", AGREGADOS / "injecao_fnr_c50.csv")
    print(f"✅ {OUT / 'injecao_fnr_c50.csv'}")


if __name__ == "__main__":
    main()
