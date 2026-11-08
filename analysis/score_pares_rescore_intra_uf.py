"""Pontuação dos pares candidatos com u medido DENTRO do bloco (UF).

`score_pares_candidatos_bimodal.py` pontua pares bloqueados por UF com o u
NACIONAL de município (0,0076). Dentro de uma UF dois pacientes diferentes
concordam no município com probabilidade muito maior, então, com o u nacional,
o peso de concordância em município sai superestimado e o de discordância
subestimado — o que inverte a ordem dos dois buckets do meio.

Aqui u é medido nos próprios pares falsos do bloco (os mesmos pares já
gravados em `_parcial_ate2025/uf=*.parquet`, sem reprocessar o BI), e os 4
buckets são re-pontuados com m nacional e, como sensibilidade, com m medido nos
pares verdadeiros. Também por UF, para mostrar a variação do u.

Saídas em dados/stats/score_pares_candidatos_bimodal/ (+ outputs/):
    distribuicao_score_intraUF_ate2025.csv, u_m_empirico_por_uf_ate2025.csv
"""
from __future__ import annotations

import math
import shutil

import duckdb
import pandas as pd

import _raiz  # noqa: F401
import paths as _paths
from score_pares_candidatos_bimodal import M_MUN, M_SEXO, U_MUN, U_SEXO

DIR = _paths.STATS / "score_pares_candidatos_bimodal"


def peso(concorda: bool, m: float, u: float) -> float:
    return math.log2(m / u) if concorda else math.log2((1 - m) / (1 - u))


def main() -> None:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT regexp_extract(filename, 'uf=(\\d+)', 1) AS uf, mun_concorda, sexo_concorda,
               par_verdadeiro, SUM(n_pares) AS n
        FROM read_parquet('{DIR}/_parcial_ate2025/*.parquet', filename=true)
        GROUP BY ALL""").df()

    def um(g: pd.DataFrame) -> pd.Series:
        f, v = g[~g.par_verdadeiro], g[g.par_verdadeiro]
        return pd.Series({
            "n_pares_falsos": f.n.sum(), "n_pares_verdadeiros": v.n.sum(),
            "u_mun": f.loc[f.mun_concorda, "n"].sum() / f.n.sum(),
            "u_sexo": f.loc[f.sexo_concorda, "n"].sum() / f.n.sum(),
            "m_mun": v.loc[v.mun_concorda, "n"].sum() / v.n.sum() if v.n.sum() else float("nan"),
            "m_sexo": v.loc[v.sexo_concorda, "n"].sum() / v.n.sum() if v.n.sum() else float("nan"),
        })
    por_uf = df.groupby("uf")[["mun_concorda", "sexo_concorda", "par_verdadeiro", "n"]].apply(um).reset_index()
    total = um(df)
    print(f"u intra-UF: mun {total.u_mun:.4f} (nacional {U_MUN:.4f}), sexo {total.u_sexo:.4f} "
          f"(nacional {U_SEXO:.4f}); m medido: mun {total.m_mun:.3f}, sexo {total.m_sexo:.3f}")

    cel = df.groupby(["mun_concorda", "sexo_concorda", "par_verdadeiro"]).n.sum().unstack(fill_value=0)
    linhas = []
    for (mc, sc), r in cel.iterrows():
        linhas.append({
            "mun_concorda": mc, "sexo_concorda": sc,
            "n_pares": int(r.sum()), "n_verdadeiros": int(r.get(True, 0)),
            "taxa_par_verdadeiro": r.get(True, 0) / r.sum(),
            "score_bits_publicado_u_nacional": peso(mc, M_MUN, U_MUN) + peso(sc, M_SEXO, U_SEXO),
            "score_bits_u_intraUF_m_nacional": peso(mc, M_MUN, total.u_mun) + peso(sc, M_SEXO, total.u_sexo),
            "score_bits_u_intraUF_m_medido": peso(mc, total.m_mun, total.u_mun) + peso(sc, total.m_sexo, total.u_sexo),
        })
    t = pd.DataFrame(linhas).sort_values("score_bits_u_intraUF_m_nacional", ascending=False)
    pub = t.sort_values("score_bits_publicado_u_nacional", ascending=False)["taxa_par_verdadeiro"]
    novo = t["taxa_par_verdadeiro"]
    t["ordem_publicada_monotona"] = bool(pub.is_monotonic_decreasing)
    t["ordem_nova_monotona"] = bool(novo.is_monotonic_decreasing)
    print(t.round(4).to_string(index=False))
    assert t["n_pares"].sum() == 243000000, t["n_pares"].sum()
    assert not pub.is_monotonic_decreasing, "com o u nacional, a inversão deveria aparecer"

    for d, nome in [(t, "distribuicao_score_intraUF_ate2025.csv"), (por_uf, "u_m_empirico_por_uf_ate2025.csv")]:
        d.to_csv(DIR / nome, index=False)
        destino = _paths.STATS.parent.parent / "outputs"
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DIR / nome, destino / nome)
        print(f"✅ {DIR / nome}")


if __name__ == "__main__":
    main()
