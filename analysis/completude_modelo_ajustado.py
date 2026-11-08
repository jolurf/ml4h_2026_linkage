"""Raça faltante × recall do linkage cego, ajustado e dentro do estabelecimento,
e ICs com cluster.

Base: os 2 783 627 pares verdadeiros do linkage cego BI×AQ, um por paciente,
enriquecidos por `completude_enriquecer_pacientes.py`. Desfecho: o par foi
recuperado (blocking em município + classificador em sexo).

1. **Efeito da raça faltante** em pontos percentuais:
   - bruto: recall(raça faltante) − recall(raça informada);
   - ajustado: logística binomial em células (raça, tem_sexo, tem_regiao, ano, UF)
     com ano e UF como efeitos fixos e sexo/região informados como covariáveis
     separadas — NÃO o escore somado, que inclui a própria raça. Diferença de
     risco por g-computação; IC por SE cluster-robusto por UF;
   - dentro do estabelecimento: diferença de risco de Mantel–Haenszel
     estratificada pelo CNES modal do paciente (variância de Greenland–Robins),
     só estratos com os dois grupos;
   - só registros "bloqueáveis" (sexo e região informados): separa o déficit
     mecânico (faltar o campo do blocking) do que acompanha a raça faltante.
2. **Cluster bootstrap** das duas diferenças de FNR do paper (Preto/Pardo vs
   Branco, 0,67 pp; Norte/Nordeste vs Sul/Sudeste, 2,30 pp): reamostra
   municípios, UFs ou CNES inteiros, 1 000 réplicas.

Saídas em dados/stats/completude_fnr_parity/ (+ outputs/):
    efeito_raca_faltante_ate2025.csv, modelo_recall_ajustado_ate2025.csv,
    paridade_fnr_cluster_ate2025.csv
"""
from __future__ import annotations

import shutil

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import _raiz  # noqa: F401
import paths as _paths

DIR = _paths.STATS / "completude_fnr_parity"
ENR = DIR / "_parcial_ate2025" / "pacientes_enriq_part=*.parquet"
AGREGADOS = _paths.STATS.parents[1] / "outputs"
N_BOOT = 1000
SEMENTE = 20261008


def gravar(df: pd.DataFrame, nome: str) -> None:
    df.to_csv(DIR / nome, index=False)
    AGREGADOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DIR / nome, AGREGADOS / nome)
    print(f"✅ {DIR / nome}")


def mantel_haenszel_rd(g: pd.DataFrame) -> tuple[float, float, float, int, int]:
    """RD de MH entre exposto (raça faltante) e não exposto por estrato.
    g: colunas estrato, a=recuperados expostos, n1=expostos, c=recuperados não expostos, n0."""
    g = g[(g.n1 > 0) & (g.n0 > 0)].copy()
    N = g.n1 + g.n0
    w = g.n1 * g.n0 / N
    rd = (g.a * g.n0 - g.c * g.n1).div(N).sum() / w.sum()
    # Greenland–Robins (1985) / Sato
    p1, p0 = g.a / g.n1, g.c / g.n0
    num = ((g.n1 ** 2 * g.n0 * p1 * (1 - p1) + g.n0 ** 2 * g.n1 * p0 * (1 - p0)) / N ** 2).sum()
    se = np.sqrt(num) / w.sum()
    return rd, rd - 1.96 * se, rd + 1.96 * se, len(g), int(g.n1.sum() + g.n0.sum())


def main() -> None:
    con = duckdb.connect()
    con.execute(f"CREATE TABLE p AS SELECT *, (raca_cor = 'Sem informação')::INT AS raca_falta FROM '{ENR}'")
    n = con.execute("SELECT COUNT(*) FROM p").fetchone()[0]
    assert n == 2_783_627, n

    # ---- 1. efeito da raça faltante ----
    bruto = con.execute("""SELECT raca_falta, AVG(recuperado::INT) r, COUNT(*) n FROM p GROUP BY 1""").df().set_index("raca_falta")
    rd_bruto = 100 * (bruto.loc[1, "r"] - bruto.loc[0, "r"])

    # Ajustado SÓ nos registros bloqueáveis (sexo e região informados). Sem
    # município o blocking é impossível e o recall nessas células é ~0: incluí-las
    # dá separação perfeita, e esse déficit é mecânico, não da raça (linha 4 abaixo).
    cel = con.execute("""
        SELECT raca_cor, ano, uf, COUNT(*) n, SUM(recuperado::INT) k
        FROM p WHERE tem_sexo = 1 AND tem_regiao = 1 GROUP BY ALL""").df()
    cel["nk"] = cel.n - cel.k
    cel["raca_cor"] = pd.Categorical(cel.raca_cor).remove_unused_categories()
    mod = smf.glm("k + nk ~ C(raca_cor, Treatment('Branca')) + C(ano) + C(uf)",
                  data=cel, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(cel.uf)[0]})
    from patsy import build_design_matrices
    inf = cel[cel.raca_cor != "Sem informação"]
    alt = inf.assign(raca_cor=pd.Categorical(["Sem informação"] * len(inf), categories=cel.raca_cor.cat.categories))
    di = mod.model.data.design_info
    Xo = np.asarray(build_design_matrices([di], inf)[0])
    Xa = np.asarray(build_design_matrices([di], alt)[0])

    def rd(b: np.ndarray) -> float:
        """g-computação: recall previsto se cada paciente com raça informada
        tivesse raça faltante, menos o previsto com a raça dele."""
        return 100 * np.average(1 / (1 + np.exp(-Xa @ b)) - 1 / (1 + np.exp(-Xo @ b)), weights=inf.n)
    rd_adj = rd(mod.params.values)
    rng = np.random.default_rng(SEMENTE)
    sims = [rd(b) for b in rng.multivariate_normal(mod.params.values, mod.cov_params().values, size=500)]
    adj_lo, adj_hi = np.percentile(sims, [2.5, 97.5])

    mh = con.execute("""
        SELECT cnes AS estrato,
               SUM(recuperado::INT) FILTER (WHERE raca_falta = 1) a, COUNT(*) FILTER (WHERE raca_falta = 1) n1,
               SUM(recuperado::INT) FILTER (WHERE raca_falta = 0) c, COUNT(*) FILTER (WHERE raca_falta = 0) n0
        FROM p WHERE cnes IS NOT NULL AND tem_sexo = 1 AND tem_regiao = 1 GROUP BY 1""").df().fillna(0)
    rd_mh, mh_lo, mh_hi, n_estr, n_cob = mantel_haenszel_rd(mh)
    # MH por CNES × ano (o ajustado tem ano; o MH só por CNES não) com IC por
    # bootstrap de ESTABELECIMENTOS inteiros — a mesma unidade de cluster do
    # resto, em vez da variância iid de Greenland–Robins.
    mha = con.execute("""
        SELECT cnes, ano, cnes || '|' || ano AS estrato,
               SUM(recuperado::INT) FILTER (WHERE raca_falta = 1) a, COUNT(*) FILTER (WHERE raca_falta = 1) n1,
               SUM(recuperado::INT) FILTER (WHERE raca_falta = 0) c, COUNT(*) FILTER (WHERE raca_falta = 0) n0
        FROM p WHERE cnes IS NOT NULL AND tem_sexo = 1 AND tem_regiao = 1 GROUP BY 1, 2, 3""").df().fillna(0)
    mha = mha[(mha.n1 > 0) & (mha.n0 > 0)].reset_index(drop=True)
    rd_mha = mantel_haenszel_rd(mha)[0]
    rng_mh = np.random.default_rng(SEMENTE)
    cnes_ids = mha.cnes.unique()
    grupos = mha.groupby("cnes").indices
    boots = []
    for _ in range(N_BOOT):
        amostra = rng_mh.choice(cnes_ids, size=len(cnes_ids), replace=True)
        idx = np.concatenate([grupos[c] for c in amostra])
        boots.append(mantel_haenszel_rd(mha.iloc[idx])[0])
    mha_lo, mha_hi = np.percentile(boots, [2.5, 97.5])

    blq = con.execute("""SELECT raca_falta, AVG(recuperado::INT) r, COUNT(*) n FROM p
                         WHERE tem_sexo = 1 AND tem_regiao = 1 GROUP BY 1""").df().set_index("raca_falta")
    rd_blq = 100 * (blq.loc[1, "r"] - blq.loc[0, "r"])

    efeito = pd.DataFrame([
        {"estimativa": "bruto", "rd_pontos": rd_bruto, "ic_lo": np.nan, "ic_hi": np.nan,
         "recall_falta": 100 * bruto.loc[1, "r"], "recall_informada": 100 * bruto.loc[0, "r"],
         "n": n, "n_estratos": np.nan, "nota": "recall(raça faltante) − recall(raça informada)"},
        {"estimativa": "ajustado por ano e UF (registros bloqueáveis)", "rd_pontos": rd_adj,
         "ic_lo": adj_lo, "ic_hi": adj_hi, "recall_falta": np.nan, "recall_informada": np.nan, "n": n,
         "n_estratos": np.nan, "nota": "logística em células, g-computação, SE cluster por UF"},
        {"estimativa": "dentro do estabelecimento (MH por CNES)", "rd_pontos": 100 * rd_mh,
         "ic_lo": 100 * mh_lo, "ic_hi": 100 * mh_hi, "recall_falta": np.nan, "recall_informada": np.nan,
         "n": n_cob, "n_estratos": n_estr, "nota": "registros bloqueáveis; estratos com os dois grupos; Greenland–Robins"},
        {"estimativa": "dentro do estabelecimento e ano (MH por CNES×ano, bootstrap por CNES)",
         "rd_pontos": 100 * rd_mha, "ic_lo": 100 * mha_lo, "ic_hi": 100 * mha_hi,
         "recall_falta": np.nan, "recall_informada": np.nan,
         "n": int(mha.n1.sum() + mha.n0.sum()), "n_estratos": len(mha),
         "nota": "registros bloqueáveis; estratos CNES×ano com os dois grupos; IC por bootstrap de CNES"},
        {"estimativa": "só registros com sexo e região informados", "rd_pontos": rd_blq,
         "ic_lo": np.nan, "ic_hi": np.nan, "recall_falta": 100 * blq.loc[1, "r"],
         "recall_informada": 100 * blq.loc[0, "r"], "n": int(blq.n.sum()), "n_estratos": np.nan,
         "nota": "remove o déficit mecânico de faltar campo do blocking"},
    ])
    print(efeito.round(3).to_string(index=False))
    gravar(efeito, "efeito_raca_faltante_ate2025.csv")
    coefs = pd.DataFrame({"termo": mod.params.index, "coef_logit": mod.params.values,
                          "se_cluster_uf": mod.bse.values, "or": np.exp(mod.params.values)})
    coefs = coefs[~coefs.termo.str.startswith(("C(ano)", "C(uf)"))]
    gravar(coefs, "modelo_recall_ajustado_ate2025.csv")

    # ---- 2. cluster bootstrap das diferenças de FNR ----
    comparacoes = {
        "Preto/Pardo vs Branco": ("raca_cor IN ('Preta','Parda')", "raca_cor = 'Branca'"),
        "Norte/Nordeste vs Sul/Sudeste": ("regiao IN ('Norte','Nordeste')", "regiao IN ('Sul','Sudeste')"),
    }
    ref = pd.read_csv(DIR / "paridade_fnr_ate2025.csv").set_index("comparacao")
    linhas = []
    rng = np.random.default_rng(SEMENTE)
    for comp, (ga, gb) in comparacoes.items():
        for unidade in ("mun_res", "uf", "cnes"):
            g = con.execute(f"""
                SELECT COALESCE({unidade}, '?') AS cl,
                       COUNT(*) FILTER (WHERE {ga}) na, SUM((NOT recuperado)::INT) FILTER (WHERE {ga}) fa,
                       COUNT(*) FILTER (WHERE {gb}) nb, SUM((NOT recuperado)::INT) FILTER (WHERE {gb}) fb
                FROM p GROUP BY 1""").df().fillna(0)
            g = g[(g.na + g.nb) > 0]  # por UF: 27 estados + o estrato '99' (UF desconhecida)
            arr = g[["na", "fa", "nb", "fb"]].to_numpy(float)
            est = arr[:, 1].sum() / arr[:, 0].sum() - arr[:, 3].sum() / arr[:, 2].sum()
            assert abs(est - ref.loc[comp, "diferenca"]) < 1e-9, (comp, est)
            idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
            s = arr[idx].sum(axis=1)
            boot = s[:, 1] / s[:, 0] - s[:, 3] / s[:, 2]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            linhas.append({"comparacao": comp, "unidade_cluster": unidade, "n_clusters": len(arr),
                           "diferenca_pp": 100 * est, "ic_lo_pp": 100 * lo, "ic_hi_pp": 100 * hi,
                           "ic_lo_iid_pp": 100 * ref.loc[comp, "ic_lo"], "ic_hi_iid_pp": 100 * ref.loc[comp, "ic_hi"],
                           "razao_largura_cluster_iid": (hi - lo) / (ref.loc[comp, "ic_hi"] - ref.loc[comp, "ic_lo"])})
    cl = pd.DataFrame(linhas)
    print(cl.round(3).to_string(index=False))
    gravar(cl, "paridade_fnr_cluster_ate2025.csv")


if __name__ == "__main__":
    main()
