"""Coleta os números do artigo ML4H direto do CSV que produziu cada um.

Nenhum número do texto é digitado: cada tag lê uma célula de um arquivo em
`outputs/`, e o JSON gravado registra tag → valor → script → arquivo → célula.
`paper/gerar_tex_final.py` substitui as tags do manuscrito e falha se sobrar alguma.

Uso: python analysis/coletar_numeros.py   (grava numbers/numbers_all.json, com
todas as tags coletadas; `numbers/make_provenance.py` restringe às tags que o
artigo usa e grava numbers/numbers.json e numbers/provenance.md)

O campo `script` é o caminho do script no pipeline dos autores (scripts/ ou
projetos/Nefroblastoma_sensibilidade/); neste repositório todos estão em
analysis/, com o mesmo nome de arquivo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
O = AQUI.parent / "outputs"
NUM: dict[str, dict] = {}


def tag(nome: str, valor, fmt: str, script: str, arquivo: str, celula: str) -> None:
    v = float(valor)
    NUM[nome] = {"valor": fmt.format(v), "bruto": v, "script": script, "arquivo": f"outputs/{arquivo}",
                 "celula": celula}


def ler(arq: str, **kw) -> pd.DataFrame:
    return pd.read_csv(O / arq, **kw)


def extras() -> None:
    """Cascata por regra e pool, Tabela 2, cluster, raça faltante, duplicatas do
    Fellegi–Sunter, trincas, calibração do C50, link RHC e orçamento de óbitos."""
    # --- cascata: absorção de R1, precisão por regra, faixas por UF, pediátrico ---
    S = "scripts/cascata_registro_retido_uf.py"
    A = "cascata_registro_retido_ate2025.csv"
    ca = ler(A)
    coorte = "1% dos pacientes (tamanho de coorte)"
    prod = "produção (R1→R3→R4)"
    sel = lambda uf, pool, ordem, regra: ca[(ca.uf == uf) & (ca.pool == pool) & (ca.ordem == ordem) & (ca.regra == regra)].iloc[0]  # noqa: E731
    r1 = sel("todas", coorte, "sozinha", "R1")
    cas = sel("todas", coorte, prod, "cascata (acumulada)")
    tag("A-REC-R1", 100 * r1.recall, "{:.1f}", S, A, "todas/coorte/sozinha/R1.recall")
    tag("A-ABS", 100 * (cas.corretas - r1.corretas) / (cas.consultas - r1.corretas), "{:.1f}", S, A,
        "(corretas cascata − corretas R1)/(consultas − corretas R1)")
    for regra in ("R1", "R3", "R4"):
        x = sel("todas", coorte, prod, regra)
        tag(f"A-PREC-{regra}", 100 * x.precisao, "{:.1f}", S, A, f"todas/coorte/produção/{regra}.precisao")
    tag("A-UNLINKED", cas.consultas - cas.vinculadas, "{:,.0f}", S, A, "consultas − vinculadas")
    por_uf = ca[(ca.uf != "todas") & (ca.pool == coorte) & (ca.ordem == prod) & (ca.regra == "cascata (acumulada)")]
    tag("A-REC-UF-MIN", 100 * por_uf.recall.min(), "{:.1f}", S, A, "min recall por UF")
    tag("A-REC-UF-MAX", 100 * por_uf.recall.max(), "{:.1f}", S, A, "max recall por UF")
    tag("A-PREC-UF-MIN", 100 * por_uf.precisao.min(), "{:.1f}", S, A, "min precisão por UF")
    tag("A-PREC-UF-MAX", 100 * por_uf.precisao.max(), "{:.1f}", S, A, "max precisão por UF")
    c50 = sel("todas", coorte, "C50 (R3→R1→R4)", "cascata (acumulada)")
    tag("A-PREC-C50ORD", 100 * c50.precisao, "{:.1f}", S, A, "ordem C50 precisão")
    ped = ca[(ca.uf == "todas") & ca.pool.str.contains("pediátrico") & (ca.ordem == prod) & (ca.regra == "cascata (acumulada)")].iloc[0]
    tag("A-REC-PED", 100 * ped.recall, "{:.1f}", S, A, "pediátrico.recall")
    tag("A-PREC-PED", 100 * ped.precisao, "{:.1f}", S, A, "pediátrico.precisao")
    tag("A-AMB-PED", 100 * ped.ambiguas / ped.vinculadas, "{:.0f}", S, A, "pediátrico ambíguas/vinculadas")
    tag("A-N-PED", ped.consultas, "{:,.0f}", S, A, "pediátrico.consultas")

    # --- Tabela 2 a três casas ---
    S = "projetos/Nefroblastoma_sensibilidade/tabela2_raca_faltante.py"
    t2 = ler("tabela2_revisada.csv").set_index("metodo")
    for met, p in [("Complete case", "F-CC"), ("IPW, P(raça observada)", "F-IPW"), ("Imputação múltipla", "F-MI")]:
        r = t2.loc[met]
        NUM[p] = {"valor": f"{r.hr_white_vs_black:.3f} [{r.hr_ic_lo:.3f}, {r.hr_ic_hi:.3f}]", "script": S,
                  "arquivo": "outputs/tabela2_revisada.csv", "celula": f"{met}.hr (ic), 3 casas"}

    # --- cluster por CNES, e o 28º estrato ---
    S = "scripts/completude_modelo_ajustado.py"
    cl = ler("paridade_fnr_cluster_ate2025.csv")
    for comp, p in [("Preto/Pardo vs Branco", "F-CI1-CNES"), ("Norte/Nordeste vs Sul/Sudeste", "F-CI2-CNES")]:
        r = cl[(cl.comparacao == comp) & (cl.unidade_cluster == "cnes")].iloc[0]
        NUM[p] = {"valor": f"[{r.ic_lo_pp:.2f}, {r.ic_hi_pp:.2f}]", "script": S,
                  "arquivo": "outputs/paridade_fnr_cluster_ate2025.csv", "celula": f"{comp}.cnes"}
    rr = cl[cl.unidade_cluster == "cnes"]
    tag("F-WIDEN-MIN", rr.razao_largura_cluster_iid.min(), "{:.0f}", S, "paridade_fnr_cluster_ate2025.csv", "min razão largura (todas unidades)")
    tag("F-WIDEN-MAX", cl.razao_largura_cluster_iid.max(), "{:.0f}", S, "paridade_fnr_cluster_ate2025.csv", "max razão largura")
    tag("F-WIDEN-MIN", cl.razao_largura_cluster_iid.min(), "{:.0f}", S, "paridade_fnr_cluster_ate2025.csv", "min razão largura")

    # --- E: MH por CNES×ano com bootstrap por CNES; sem sinal ---
    e = ler("efeito_raca_faltante_ate2025.csv").set_index("estimativa")
    r = e.loc["dentro do estabelecimento e ano (MH por CNES×ano, bootstrap por CNES)"]
    NUM["E-FAC"] = {"valor": f"{-r.rd_pontos:.1f} [{-r.ic_hi:.1f}, {-r.ic_lo:.1f}]", "script": S,
                    "arquivo": "outputs/efeito_raca_faltante_ate2025.csv", "celula": "MH CNES×ano (sinal invertido: déficit)"}
    tag("E-FAC-N", r.n_estratos, "{:,.0f}", S, "efeito_raca_faltante_ate2025.csv", "MH CNES×ano.n_estratos")
    r = e.loc["ajustado por ano e UF (registros bloqueáveis)"]
    NUM["E-ADJ"] = {"valor": f"{-r.rd_pontos:.1f} [{-r.ic_hi:.1f}, {-r.ic_lo:.1f}]", "script": S,
                    "arquivo": "outputs/efeito_raca_faltante_ate2025.csv", "celula": "ajustado (sinal invertido: déficit)"}
    tag("E-RAW", -e.loc["bruto", "rd_pontos"], "{:.1f}", S, "efeito_raca_faltante_ate2025.csv", "bruto (déficit)")
    tag("E-BLQ", -e.loc["só registros com sexo e região informados", "rd_pontos"], "{:.1f}", S,
        "efeito_raca_faltante_ate2025.csv", "bloqueáveis (déficit)")
    tag("E-REC-FALTA", e.loc["bruto", "recall_falta"], "{:.1f}", S, "efeito_raca_faltante_ate2025.csv", "bruto.recall_falta")
    tag("E-REC-INF", e.loc["bruto", "recall_informada"], "{:.1f}", S, "efeito_raca_faltante_ate2025.csv", "bruto.recall_informada")

    # --- FS: faixa e RHC_ID ---
    S = "scripts/fs_diagnostico_u_m.py"
    sd = ler("fs_u_sem_duplicatas_coorte.csv").set_index("duplicatas_removidas_por")
    tag("D-RATIO-ANO", sd.loc["mesmo_cnes_ano", "razao"], "{:.1f}", S, "fs_u_sem_duplicatas_coorte.csv", "mesmo_cnes_ano.razao")
    dup = ler("fs_duplicatas_resumo_coorte.csv").set_index("criterio")
    tag("D-DUP-RHC", dup.loc["mesmo registro de tumor no RHC (duplicata definitiva)", "n_pares"], "{:,.0f}", S,
        "fs_duplicatas_resumo_coorte.csv", "mesmo RHC_ID.n_pares")

    # --- trincas distintas observadas × esperadas ---
    S = "scripts/colisao_trinca_baseline_acaso.py"
    b = ler("colisao_trinca_baseline_ate2025.csv").iloc[0]
    tag("F-TRINCAS-OBS", b.trincas_observadas / 1e6, "{:.1f}", S, "colisao_trinca_baseline_ate2025.csv", "trincas_observadas (mi)")
    tag("F-TRINCAS-ESP", b.trincas_esperadas_acaso / 1e6, "{:.1f}", S, "colisao_trinca_baseline_ate2025.csv", "trincas_esperadas (mi)")

    # --- C50: calibração por decis ---
    S = "scripts/c50_calibracao_rotulos.py"
    gl = ler("calibracao_rotulos_global.csv").set_index(["rotulo_treino", "rotulo_avaliacao"])
    for tr in ("SIM", "RHC", "Uniao"):
        for av in ("SIM", "RHC", "Uniao"):
            tag(f"C-DEC-{tr}-{av}", gl.loc[(tr, av), "cal_decil_mae_pp"], "{:.1f}", S,
                "calibracao_rotulos_global.csv", f"{tr}->{av}.cal_decil_mae_pp")
    sb = ler("calibracao_rotulos_subgrupos.csv")
    r = sb[(sb.rotulo_treino == "RHC") & (sb.rotulo_avaliacao == "Uniao") & (sb.grupo == "Preta")].iloc[0]
    tag("C-SUB", -r.calibracao_na_media_pp, "{:.1f}", S, "calibracao_rotulos_subgrupos.csv", "RHC->Uniao.Preta (subestimação)")

    # --- RHC: validação com gabarito e sensibilidade por qualidade do match ---
    S = "projetos/Nefroblastoma_sensibilidade/validacao_rhc_surrogate.py"
    v = ler("validacao_rhc_resumo.csv").iloc[0]
    for col, p in [("cobertura", "RHC-COB"), ("precisao", "RHC-PREC"), ("fdr", "RHC-FDR"), ("recall", "RHC-REC")]:
        tag(p, 100 * v[col], "{:.1f}", S, "validacao_rhc_resumo.csv", col)
    tag("RHC-SONDAS", v.sondas, "{:,.0f}", S, "validacao_rhc_resumo.csv", "sondas")
    S = "projetos/Nefroblastoma_sensibilidade/sensibilidade_match_rhc.py"
    m = ler("tabela_sensibilidade_match_rhc.csv").set_index("estrato")
    st = m.loc["CNES ∧ ≤ 7 dias ∧ mesmo ano de nascimento"]
    tag("RHC-STRICT-N", st.n, "{:,.0f}", S, "tabela_sensibilidade_match_rhc.csv", "estrito.n")
    tag("RHC-STRICT-OS", st.os_5a_pct, "{:.1f}", S, "tabela_sensibilidade_match_rhc.csv", "estrito.os")
    tag("RHC-STRICT-HR", st.hr_estadio_iv, "{:.2f}", S, "tabela_sensibilidade_match_rhc.csv", "estrito.hr")
    cn = m.loc["match pelo CNES do hospital (não por município)"]
    tag("RHC-CNES-N", cn.n, "{:,.0f}", S, "tabela_sensibilidade_match_rhc.csv", "CNES.n")
    tag("RHC-CNES-OS", cn.os_5a_pct, "{:.1f}", S, "tabela_sensibilidade_match_rhc.csv", "CNES.os")
    tag("RHC-CNES-HR", cn.hr_estadio_iv, "{:.2f}", S, "tabela_sensibilidade_match_rhc.csv", "CNES.hr")
    estr = m[~m.index.str.startswith("complemento") & (m.index != "todos (referência)")]
    tag("RHC-OS-MIN", estr.os_5a_pct.min(), "{:.1f}", S, "tabela_sensibilidade_match_rhc.csv", "min OS estratos estritos")
    tag("RHC-OS-MAX", estr.os_5a_pct.max(), "{:.1f}", S, "tabela_sensibilidade_match_rhc.csv", "max OS estratos estritos")

    # --- orçamento de erro dos óbitos (5 anos) ---
    S = "projetos/Nefroblastoma_sensibilidade/concordancia_rhc_sus.py"
    f5 = ler("tabela_obitos_c1_por_fonte.csv").query("horizonte == '5 anos'").set_index("origem")
    for o, p in [("so_SIA", "OB-SIA"), ("so_SIH", "OB-SIH"), ("ambos", "OB-AMBOS")]:
        tag(p, f5.loc[o, "n"], "{:,.0f}", S, "tabela_obitos_c1_por_fonte.csv", f"5 anos {o}.n")
        tag(p + "-CONF", f5.loc[o, "confirmados_rhc"], "{:,.0f}", S, "tabela_obitos_c1_por_fonte.csv", f"5 anos {o}.confirmados")
        tag(p + "-PCT", f5.loc[o, "pct_confirmados"], "{:.1f}", S, "tabela_obitos_c1_por_fonte.csv", f"5 anos {o}.pct")
    tag("SIH-CONF-PCT", f5.loc["so_SIH", "pct_confirmados"], "{:.1f}", S, "tabela_obitos_c1_por_fonte.csv", "5 anos so_SIH.pct")
    tag("SIH-ONLY", f5.loc["so_SIH", "n"], "{:,.0f}", S, "tabela_obitos_c1_por_fonte.csv", "5 anos so_SIH.n")
    rx = ler("tabela_rhc_x_sus.csv").set_index("horizonte").loc["5 anos"]
    tag("OB-RHCONLY", rx.so_rhc, "{:,.0f}", S, "tabela_rhc_x_sus.csv", "5 anos so_rhc")
    tag("RX-AMBOS", ler("tabela_rhc_x_sus.csv").set_index("horizonte").loc["completo", "ambos"], "{:,.0f}", S, "tabela_rhc_x_sus.csv", "completo.ambos")
    tag("RX-SORHC", ler("tabela_rhc_x_sus.csv").set_index("horizonte").loc["completo", "so_rhc"], "{:,.0f}", S, "tabela_rhc_x_sus.csv", "completo.so_rhc")
    tag("RX-SOPIPE", ler("tabela_rhc_x_sus.csv").set_index("horizonte").loc["completo", "so_pipeline"], "{:,.0f}", S, "tabela_rhc_x_sus.csv", "completo.so_pipeline")
    S = "projetos/Nefroblastoma_sensibilidade/propagacao_erro_linkage.py"
    cj = ler("propagacao_conjuntos.csv")["n"].tolist()
    tag("OB-SIM", cj[4], "{:,.0f}", S, "propagacao_conjuntos.csv", "C2 em 5 anos")
    tag("OB-SIM-CONF", cj[4] - cj[1], "{:,.0f}", S, "propagacao_conjuntos.csv", "C2 − S_SIM")
    tag("OB-SIM-PCT", 100 * (cj[4] - cj[1]) / cj[4], "{:.1f}", S, "propagacao_conjuntos.csv", "(C2 − S_SIM)/C2")
    g = ler("propagacao_gap_racial.csv")
    gm = g.iloc[(g.fnr_branca - 0.025).abs().argmin()]
    NUM["B-GAP"] = {"valor": f"{gm.gap_med:.1f} [{gm.gap_p025:.1f}, {gm.gap_p975:.1f}]", "script": S,
                    "arquivo": "outputs/propagacao_gap_racial.csv", "celula": "fnr_branca=0.025 med [p2.5, p97.5]"}
    lim = ler("propagacao_limiares.csv").set_index("deslocamento_os_pp")
    tag("B-HALFPT", 100 * lim.loc[0.5, "perda_necessaria"], "{:.0f}", S, "propagacao_limiares.csv", "0.5.perda_necessaria")


def extras2() -> None:
    """u condicional, blocking, recall por raça, entropia, cascata, RHC e C50."""
    # --- razão condicional u(nasc+sexo+mun)/u(nasc) ---
    S = "scripts/fs_diagnostico_u_m.py"
    u = ler("fs_u_corrigido_coorte.csv").set_index("chave")
    k3 = "nasc+sexo+mun_res"
    for c, p in [("u_com_autopar", "D-COND-OLD"), ("u_sem_autopar", "D-COND-NEW")]:
        tag(p, u.loc[k3, c] / u.loc["data_nascimento", c], "{:.2f}", S, "fs_u_corrigido_coorte.csv", f"{k3}/{c} ÷ data_nascimento")
    tag("D-COND-IND", u.loc["sexo", "u_sem_autopar"] * u.loc["mun_res", "u_sem_autopar"], "{:.3f}", S,
        "fs_u_corrigido_coorte.csv", "u_sexo × u_mun (sem auto-par)")
    # --- precisão do blocking do linkage cego BI×AQ; recall por raça informada ---
    S = "scripts/completude_fnr_parity.py"
    m = ler("metricas_por_estrato_ate2025.csv").set_index("estrato")
    pb = m.loc["Nacional", "precisao_bloco_nacional"]
    NUM["BLK-PREC"] = {"valor": f"{pb:.2e}".replace("e-08", "×10⁻⁸"), "script": S,
                       "arquivo": "outputs/metricas_por_estrato_ate2025.csv", "celula": "Nacional.precisao_bloco_nacional"}
    tag("BLK-1IN", 1 / pb / 1e6, "{:.1f}", S, "metricas_por_estrato_ate2025.csv", "1/precisão (milhões)")
    for r, p in [("Branca", "REC-BRANCA"), ("Preta", "REC-PRETA"), ("Parda", "REC-PARDA"), ("Amarela", "REC-AMARELA"),
                 ("Indígena", "REC-INDIGENA"), ("Sem informação", "REC-SI")]:
        tag(p, 100 * m.loc[f"Raça/cor: {r}", "recall"], "{:.1f}", S, "metricas_por_estrato_ate2025.csv", f"Raça/cor: {r}.recall")
    tag("REC-N-NAC", m.loc["Nacional", "n"], "{:,.0f}", S, "metricas_por_estrato_ate2025.csv", "Nacional.n")
    S = "scripts/tabela1_cns_nacional.py"
    t1 = pd.read_csv(O / "tabela1_nacional_ate2025.csv")
    tag("REG-TOTAL", t1[t1.dimensao == "sexo"].total.sum() / 1e9, "{:.1f}", S, "tabela1_nacional_ate2025.csv", "Σ total (sexo), bilhões")
    tag("REG-CONF", 100 * t1[t1.dimensao == "sexo"].com_cns.sum() / t1[t1.dimensao == "sexo"].total.sum(), "{:.2f}", S,
        "tabela1_nacional_ate2025.csv", "Σ com_cns / Σ total")
    # --- estabilidade cai com nº de registros ---
    S = "scripts/entropia_miller_madow_decomposicao.py"
    e = ler("entropia_miller_madow_ate2025.csv").set_index("faixa")
    tag("MULTI-2A4", e.loc["2-4", "pct_multi_identidade"], "{:.1f}", S, "entropia_miller_madow_ate2025.csv", "2-4.pct_multi")
    tag("MULTI-100", e.loc["100+", "pct_multi_identidade"], "{:.1f}", S, "entropia_miller_madow_ate2025.csv", "100+.pct_multi")
    # --- cascata: o que R3/R4 fazem com o que R1 deixa SEM vínculo (R1 errado não é revisitado) ---
    S = "scripts/cascata_registro_retido_uf.py"
    A = "cascata_registro_retido_ate2025.csv"
    ca = ler(A)
    q = lambda ordem, regra, pool="1% dos pacientes (tamanho de coorte)": ca[(ca.uf == "todas") & (ca.pool == pool) & (ca.ordem == ordem) & (ca.regra == regra)].iloc[0]  # noqa: E731
    r1 = q("sozinha", "R1")
    pr3, pr4 = q("produção (R1→R3→R4)", "R3"), q("produção (R1→R3→R4)", "R4")
    sem_r1 = r1.consultas - r1.vinculadas
    tag("A-UNL-R1", sem_r1, "{:,.0f}", S, A, "consultas − vinculadas por R1")
    tag("A-ABS", 100 * (pr3.corretas + pr4.corretas) / sem_r1, "{:.1f}", S, A, "(corretas R3+R4 na cascata) / sem vínculo por R1")
    tag("A-WRONG-R1", r1.vinculadas - r1.corretas, "{:,.0f}", S, A, "R1 vinculadas − corretas")
    tag("A-WRONG-R1-PCT", 100 * (r1.vinculadas - r1.corretas) / r1.vinculadas, "{:.1f}", S, A, "R1 erradas / vinculadas")
    uf = q("produção (R1→R3→R4)", "cascata (acumulada)", "UF inteira")
    tag("A-REC-UF", 100 * uf.recall, "{:.1f}", S, A, "UF inteira.recall")
    c50 = q("C50 (R3→R1→R4)", "cascata (acumulada)")
    tag("A-REC-C50ORD", 100 * c50.recall, "{:.1f}", S, A, "ordem C50.recall")
    # --- RHC: horizonte de 5 anos, rota por município, faixa de HR nos estratos estritos ---
    S = "projetos/Nefroblastoma_sensibilidade/concordancia_rhc_sus.py"
    r5 = ler("tabela_rhc_x_sus.csv").set_index("horizonte").loc["5 anos"]
    tag("RX-AMBOS", r5.ambos, "{:,.0f}", S, "tabela_rhc_x_sus.csv", "5 anos.ambos")
    tag("RX-SORHC", r5.so_rhc, "{:,.0f}", S, "tabela_rhc_x_sus.csv", "5 anos.so_rhc")
    tag("RX-SOPIPE", r5.so_pipeline, "{:,.0f}", S, "tabela_rhc_x_sus.csv", "5 anos.so_pipeline")
    NUM["A-RHC"] = {"valor": f"{r5.pct_rhc_achados_pelo_pipeline:.1f}% [{r5.ic_lo:.1f}, {r5.ic_hi:.1f}]", "script": S,
                    "arquivo": "outputs/tabela_rhc_x_sus.csv", "celula": "5 anos.pct (Wilson)"}
    f5 = ler("tabela_obitos_c1_por_fonte.csv").query("horizonte == '5 anos'").set_index("origem")
    NUM["OB-SIH-CI"] = {"valor": f"[{f5.loc['so_SIH', 'ic_lo']:.1f}, {f5.loc['so_SIH', 'ic_hi']:.1f}]", "script": S,
                        "arquivo": "outputs/tabela_obitos_c1_por_fonte.csv", "celula": "5 anos so_SIH ic"}
    S = "projetos/Nefroblastoma_sensibilidade/validacao_rhc_surrogate.py"
    v = ler("validacao_rhc_resumo.csv").iloc[0]
    tag("RHC-MUN-N", v.n_bloco_mun, "{:,.0f}", S, "validacao_rhc_resumo.csv", "n_bloco_mun")
    tag("RHC-MUN-OK", v.n_bloco_mun * v.precisao_bloco_mun, "{:,.0f}", S, "validacao_rhc_resumo.csv", "n_bloco_mun × precisão")
    S = "projetos/Nefroblastoma_sensibilidade/sensibilidade_match_rhc.py"
    mm = ler("tabela_sensibilidade_match_rhc.csv").set_index("estrato")
    estr = mm[~mm.index.str.startswith("complemento") & (mm.index != "todos (referência)")]
    tag("RHC-HR-MIN", estr.hr_estadio_iv.min(), "{:.2f}", S, "tabela_sensibilidade_match_rhc.csv", "min HR estratos estritos")
    tag("RHC-HR-MAX", estr.hr_estadio_iv.max(), "{:.2f}", S, "tabela_sensibilidade_match_rhc.csv", "max HR estratos estritos")
    st = mm.loc["CNES ∧ ≤ 7 dias ∧ mesmo ano de nascimento"]
    NUM["RHC-STRICT-OS-CI"] = {"valor": f"[{st.os_ic_lo:.1f}, {st.os_ic_hi:.1f}]", "script": S,
                               "arquivo": "outputs/tabela_sensibilidade_match_rhc.csv", "celula": "estrito.os ic"}
    cn = mm.loc["complemento: match pelo CNES do hospital (não por município)"]
    tag("RHC-MUNROUTE-N", cn.n, "{:,.0f}", S, "tabela_sensibilidade_match_rhc.csv", "complemento CNES.n")
    # --- incerteza amostral da referência (para rotular as faixas de Monte Carlo) ---
    S = "projetos/Nefroblastoma_sensibilidade/cenarios_degradacao_linkage.py"
    c0 = ler("tabela_cenarios_degradacao.csv").set_index("cenario").loc["0_referencia"]
    NUM["S0-OS-CI"] = {"valor": f"[{c0.os_5a_ic_lo:.1f}, {c0.os_5a_ic_hi:.1f}]", "script": S,
                       "arquivo": "outputs/tabela_cenarios_degradacao.csv", "celula": "0_referencia.os ic"}
    NUM["S0-HR-CI"] = {"valor": f"[{c0.hr_ic_lo:.2f}, {c0.hr_ic_hi:.2f}]", "script": S,
                       "arquivo": "outputs/tabela_cenarios_degradacao.csv", "celula": "0_referencia.hr ic"}
    S = "projetos/Nefroblastoma_sensibilidade/tabela2_raca_faltante.py"
    cc = ler("tabela2_revisada.csv").set_index("metodo").loc["Complete case"]
    NUM["GAP-CI"] = {"valor": f"[{cc.gap_ic_lo:.1f}, {cc.gap_ic_hi:.1f}]", "script": S,
                     "arquivo": "outputs/tabela2_revisada.csv", "celula": "CC.gap ic"}
    # --- C50: HR da raça parda com IC ---
    S = "scripts/c50_calibracao_rotulos.py"
    h = ler("calibracao_rotulos_hr.csv").set_index("covariavel").loc["raca_label_Parda"]
    for rot in ("SIM", "RHC", "Uniao"):
        NUM[f"C-HR-PARDA-{rot}"] = {"valor": f"{h[f'HR_{rot}']:.2f} [{h[f'ic95_inf_{rot}']:.2f}, {h[f'ic95_sup_{rot}']:.2f}]",
                                    "script": S, "arquivo": "outputs/calibracao_rotulos_hr.csv", "celula": f"raca_label_Parda.{rot} (ic)"}
    pop = ler("calibracao_rotulos_populacao.csv").set_index("passo")["n"]
    tag("C-OBS-UNION-N", pop.loc["óbitos (Uniao), população analisada"], "{:,.0f}", "scripts/c50_calibracao_rotulos.py",
        "calibracao_rotulos_populacao.csv", "óbitos (Uniao)")
    # --- por que o HR da União fica entre SIM e RHC ---
    if (O / "obitos_por_fonte_estadio.csv").exists():
        S = "scripts/c50_obitos_por_fonte_estadio.py"
        fe = ler("obitos_por_fonte_estadio.csv").set_index("estadio")
        A = "obitos_por_fonte_estadio.csv"
        tag("U-N-SIM", fe.iloc[0].n_total_so_SIM + fe.iloc[0].n_total_ambos, "{:,.0f}", S, A, "só SIM + ambos")
        tag("U-N-RHC", fe.iloc[0].n_total_so_RHC + fe.iloc[0].n_total_ambos, "{:,.0f}", S, A, "só RHC + ambos")
        tag("U-N-UNION", fe.iloc[0].n_total_so_SIM + fe.iloc[0].n_total_so_RHC + fe.iloc[0].n_total_ambos, "{:,.0f}", S, A, "soma dos três")
        tag("U-N-SORHC", fe.iloc[0].n_total_so_RHC, "{:,.0f}", S, A, "n_total_so_RHC")
        for st in ("I", "IV"):
            for rot in ("SIM", "RHC", "Uniao"):
                tag(f"U-MORT-{rot}-{st}", fe.loc[st, f"mortalidade_{rot}"], "{:.1f}", S, A, f"{st}.mortalidade_{rot}")
            for k in ("so_SIM", "so_RHC"):
                tag(f"U-{k.upper().replace('_', '')}-{st}", fe.loc[st, f"pct_{k}"], "{:.1f}", S, A, f"{st}.pct_{k}")
        for rot in ("SIM", "RHC", "Uniao"):
            tag(f"U-RATIO-{rot}", fe.loc["razão IV/I", f"mortalidade_{rot}"], "{:.1f}", S, A, f"razão IV/I mortalidade_{rot}")
        tag("U-SORHC-MIN", fe.loc[["I", "II", "III", "IV"], "pct_so_RHC"].min(), "{:.1f}", S, A, "min pct_so_RHC")
        tag("U-SORHC-MAX", fe.loc[["I", "II", "III", "IV"], "pct_so_RHC"].max(), "{:.1f}", S, A, "max pct_so_RHC")
    # --- C50: erro medido injetado no modelo treinado ---
    if (O / "injecao_fnr_c50.csv").exists():
        S = "scripts/c50_injecao_fnr.py"
        x = ler("injecao_fnr_c50.csv").set_index("cenario")
        for cen, p in [("referência (União, sem perda)", "C-INJ0"), ("FNR medida por raça", "C-INJ1"),
                       ("5× a FNR medida (estresse)", "C-INJ5")]:
            r = x.loc[cen]
            tag(f"{p}-LOST", r.obitos_perdidos, "{:,.0f}", S, "injecao_fnr_c50.csv", f"{cen}.obitos_perdidos")
            for c, nome in [("hr_estadio_iv", "IV"), ("hr_parda", "PARDA"), ("hr_preta", "PRETA"),
                            ("cal_decil_mae_pp_vs_completo", "CAL")]:
                fmt = {"CAL": "{:.1f}", "IV": "{:.2f}"}.get(nome, "{:.3f}")
                tag(f"{p}-{nome}", r[f"{c}_med"], fmt, S, "injecao_fnr_c50.csv", f"{cen}.{c}_med")


def extras3() -> None:
    """Discordância entre fontes de óbito: captura-recaptura, decomposição dos só-RHC,
    perdas da cascata e implicada, e o mesmo teste no C50."""
    S = "projetos/Nefroblastoma_sensibilidade/captura_recaptura_obitos.py"
    A = "captura_recaptura_obitos.csv"
    if not (O / A).exists():
        return
    cr = ler(A)
    t = cr.iloc[0]
    tag("CR-N", t.N_chapman, "{:.0f}", S, A, "total.N_chapman")
    tag("CR-N-LO", t.N_ic_lo, "{:.0f}", S, A, "total.N_ic_lo")
    tag("CR-N-HI", t.N_ic_hi, "{:.0f}", S, A, "total.N_ic_hi")
    tag("CR-MISSED", t.faltantes, "{:.0f}", S, A, "total.faltantes")
    tag("CR-MISSED-LO", t.faltantes_ic_lo, "{:.0f}", S, A, "total.faltantes_ic_lo")
    tag("CR-MISSED-HI", t.faltantes_ic_hi, "{:.0f}", S, A, "total.faltantes_ic_hi")
    somas = cr[cr.estrato == "soma dos estratos"].faltantes
    tag("CR-MISSED-STRAT-MIN", somas.min(), "{:.0f}", S, A, "min soma dos estratos")
    tag("CR-MISSED-STRAT-MAX", somas.max(), "{:.0f}", S, A, "max soma dos estratos")
    tag("CR-CAP-PIPE", 100 * t.captura_pipeline, "{:.0f}", S, A, "total.captura_pipeline")
    tag("CR-CAP-RHC", 100 * t.captura_rhc, "{:.0f}", S, A, "total.captura_rhc")
    tag("CR-CAP-REF", 100 * t.captura_uniao, "{:.0f}", S, A, "total.captura_uniao")
    ign = cr[(cr.eixo == "estadio") & (cr.estrato == "Ignorado")].iloc[0]
    tag("CR-CAP-IGN", 100 * ign.captura_uniao, "{:.0f}", S, A, "estádio Ignorado.captura_uniao")
    aj = cr[cr.estrato == "total, sem os prováveis falsos do RHC"].iloc[0]
    tag("CR-MISSED-AJ", aj.faltantes, "{:.0f}", S, A, "ajustado.faltantes")
    tag("CR-FALSOS", aj.so_rhc_falsos_estimados, "{:.0f}", S, A, "ajustado.so_rhc_falsos_estimados")
    tag("CR-BASE-NF", 100 * aj.taxa_nao_encontrado_base, "{:.0f}", S, A, "taxa de não encontrados entre os ambos")
    A = "captura_recaptura_cenarios.csv"
    c = ler(A).set_index("cenario")
    for cen, p in [("faltantes por estádio (principal)", "CR-MISS"), ("só remover os prováveis falsos do RHC", "CR-FALSE"),
                   ("remover os prováveis falsos do RHC e acrescentar os faltantes ajustados", "CR-BOTH")]:
        r = c.loc[cen]
        tag(f"{p}-K", abs(r.k_total), "{:.0f}", S, A, f"{cen}.k_total")
        tag(f"{p}-OS", r.os5_med, "{:.1f}", S, A, f"{cen}.os5_med")
        tag(f"{p}-HR", r.hr_med, "{:.2f}", S, A, f"{cen}.hr_med")
        tag(f"{p}-GAP", r.gap_med, "{:.1f}", S, A, f"{cen}.gap_med")
    A = "decomposicao_so_rhc.csv"
    dz = ler(A).set_index(["grupo", "classe"])["n"]
    for cls, p in [("não encontrado no SIM (provável vínculo RHC↔SUS falso ou data divergente)", "SORHC-NOTSIM"),
                   ("encontrado no SIM (qualquer classe)", "SORHC-INSIM"),
                   ("encontrado; nenhum CID do paciente igual a um campo da DO (filtro semântico)", "SORHC-SEM"),
                   ("encontrado; município de residência difere (bloqueio por município)", "SORHC-MUN"),
                   ("encontrado; passa nos três filtros (óbito antes do último registro no SUS, ou outra)", "SORHC-OTHER"),
                   ("entrou na coorte pela rota de município do link RHC", "MUNROUTE-SORHC")]:
        tag(p, dz.get(("so_rhc", cls), 0), "{:,.0f}", S, A, f"so_rhc.{cls}")
    tag("BOTH-INSIM-PCT", 100 * dz[("ambos", "encontrado no SIM (qualquer classe)")] / dz[("ambos", "TOTAL")], "{:.0f}", S, A,
        "ambos: encontrados / total")
    S = "projetos/Nefroblastoma_sensibilidade/propagacao_erro_linkage.py"
    A = "propagacao_taxas_medidas.csv"
    tx = ler(A).set_index("cenario")
    for cen, p in [("perda da cascata com CNS escondido", "B-CASC"), ("perda do pipeline implicada pela captura-recaptura", "B-CRL")]:
        if cen in tx.index:
            r = tx.loc[cen]
            tag(f"{p}-RATE", 100 * r.taxa_uniforme, "{:.1f}", S, A, f"{cen}.taxa_uniforme")
            tag(f"{p}-OS", r.os5_med, "{:.1f}", S, A, f"{cen}.os5_med")
            tag(f"{p}-HR", r.hr_med, "{:.2f}", S, A, f"{cen}.hr_med")
    S = "scripts/c50_decomposicao_so_rhc.py"
    A = "decomposicao_so_rhc_c50.csv"
    if (O / A).exists():
        z = ler(A)
        so = z[(z.grupo == "so_rhc") & (z.estadio == "todos")].iloc[0]
        tag("C50-SORHC-N", so.n, "{:,.0f}", S, A, "so_rhc.n")
        tag("C50-SORHC-INSIM", so.achados_sim, "{:,.0f}", S, A, "so_rhc.achados_sim")
        tag("C50-SORHC-INSIM-PCT", so.pct_achados_sim, "{:.0f}", S, A, "so_rhc.pct_achados_sim")
        for st in ("I", "IV"):
            tag(f"C50-SORHC-INSIM-{st}", z[(z.grupo == "so_rhc") & (z.estadio == st)].iloc[0].pct_achados_sim, "{:.0f}", S, A, f"so_rhc.{st}")
        tag("C50-BOTH-INSIM", z[(z.grupo == "ambos") & (z.estadio == "todos")].iloc[0].pct_achados_sim, "{:.0f}", S, A, "ambos.pct")
        r = z[z.grupo == "RESUMO"].reset_index(drop=True)
        tag("C50-FALSOS", r.loc[1, "n"], "{:,.0f}", S, A, "prováveis falsos")
        tag("C50-FALSOS-PCT", r.loc[1, "pct_achados_sim"], "{:.0f}", S, A, "prováveis falsos %")
        tag("C50-CR-N", r.loc[2, "n"], "{:,.0f}", S, A, "Lincoln-Petersen")
        tag("C50-CR-MORT", r.loc[2, "pct_achados_sim"], "{:.0f}", S, A, "mortalidade implícita")


def extras4() -> None:
    """Óbitos falsos do RHC como limite superior, perda mínima do link SIM, C50
    com o rótulo do pipeline separado em C1/C2 e a injeção restrita aos óbitos
    do link."""
    S = "projetos/Nefroblastoma_sensibilidade/captura_recaptura_obitos.py"
    dz = ler("decomposicao_so_rhc.csv").set_index(["grupo", "classe"])["n"]
    achados = dz[("so_rhc", "encontrado no SIM (qualquer classe)")]
    S2 = "projetos/Nefroblastoma_sensibilidade/propagacao_erro_linkage.py"
    c2 = ler("propagacao_conjuntos.csv")["n"].tolist()[4]
    tag("SIMLINK-LOSS-MIN", 100 * achados / (achados + c2), "{:.0f}", S, "decomposicao_so_rhc.csv",
        "só-RHC achados no SIM / (achados + óbitos C2 em 5 anos de propagacao_conjuntos.csv)")
    tag("SIMLINK-MISSED-DEN", achados + c2, "{:,.0f}", S, "decomposicao_so_rhc.csv", "achados + C2")
    S = "projetos/Nefroblastoma_sensibilidade/cenarios_degradacao_linkage.py"
    c = ler("tabela_cenarios_degradacao.csv").set_index("cenario")
    tag("S1-LOST", c.loc["0_referencia", "obitos"] - c.loc["1_claims_ingenuo", "obitos"], "{:,.0f}", S,
        "tabela_cenarios_degradacao.csv", "S0.obitos − S1.obitos")
    S = "projetos/Nefroblastoma_sensibilidade/sensibilidade_match_rhc.py"
    st = ler("tabela_sensibilidade_match_rhc.csv").set_index("estrato").loc["CNES ∧ ≤ 7 dias ∧ mesmo ano de nascimento"]
    tag("RHC-STRICT-DEATHS", st.obitos, "{:,.0f}", S,
        "tabela_sensibilidade_match_rhc.csv", "estrito.obitos")

    # --- C50: falsos por estádio (taxa de acerto da chave por estádio, entre os 'ambos') ---
    S = "scripts/c50_decomposicao_so_rhc.py"
    A = "decomposicao_so_rhc_c50.csv"
    z = ler(A)
    for st in ("I", "IV"):
        so = z[(z.grupo == "so_rhc") & (z.estadio == st)].iloc[0]
        b = z[(z.grupo == "ambos") & (z.estadio == st)].iloc[0].pct_achados_sim / 100
        tag(f"C50-FALSOS-PCT-{st}", 100 * max(0.0, so.n - so.achados_sim / b) / so.n, "{:.0f}", S, A,
            f"(n − achados/acerto_ambos) / n, estádio {st}")
        tag(f"C50-BOTH-INSIM-{st}", 100 * b, "{:.0f}", S, A, f"ambos.{st}.pct")
    tag("C50-BASE-NF", z[z.grupo == "RESUMO"].iloc[0].pct_achados_sim, "{:.0f}", S, A, "não-encontrados entre os ambos")
    tag("C50-FALSOS-SHARE-RHC", 100 * z[z.grupo == "RESUMO"].iloc[1].n
        / ler("calibracao_rotulos_populacao.csv").set_index("passo").loc["óbitos (RHC), população analisada", "n"],
        "{:.0f}", S, A, "prováveis falsos / óbitos com rótulo RHC")

    # --- C50: rótulo do pipeline por camada ---
    S = "scripts/c50_obitos_por_fonte_estadio.py"
    A = "obitos_pipeline_por_camada.csv"
    cm = ler(A).set_index("estadio")
    tot = cm.loc["todos"]
    tag("C50-C1", tot.obitos_c1, "{:,.0f}", S, A, "todos.obitos_c1")
    tag("C50-C2", tot.obitos_c2, "{:,.0f}", S, A, "todos.obitos_c2")
    tag("C50-C1-PCT", 100 * tot.obitos_c1 / (tot.obitos_c1 + tot.obitos_c2), "{:.0f}", S, A, "c1/(c1+c2)")
    tag("C50-C1-CONF", 100 * tot.c1_confirmados_rhc / tot.obitos_c1, "{:.0f}", S, A, "c1 confirmados/c1")
    tag("C50-C2-CONF", 100 * tot.c2_confirmados_rhc / tot.obitos_c2, "{:.0f}", S, A, "c2 confirmados/c2")
    for k in ("c1", "c2"):
        tag(f"C50-RATIO-{k.upper()}", cm.loc["IV", f"mortalidade_{k}"] / cm.loc["I", f"mortalidade_{k}"], "{:.1f}", S, A,
            f"mortalidade_{k} IV/I")

    # --- C50: injeção restrita ao link SIM (C2) ---
    S = "scripts/c50_injecao_fnr.py"
    A = "injecao_fnr_c50.csv"
    x = ler(A).set_index("cenario")
    tag("C-INJ-ELIG", x.iloc[0].obitos_elegiveis, "{:,.0f}", S, A, "obitos_elegiveis")
    r = x.loc["perda da cascata com CNS escondido"]
    tag("C-INJC-LOST", r.obitos_perdidos, "{:,.0f}", S, A, "cascata.obitos_perdidos")
    tag("C-INJC-IV", r.hr_estadio_iv_med, "{:.2f}", S, A, "cascata.hr_estadio_iv_med")
    tag("C-INJC-CAL", r.cal_decil_mae_pp_vs_completo_med, "{:.1f}", S, A, "cascata.cal")

    # --- C50: calibração diferencial por raça e HR regional ---
    S = "scripts/c50_calibracao_rotulos.py"
    sb = ler("calibracao_rotulos_subgrupos.csv")
    for tr in ("SIM", "RHC"):
        for g, p in [("Preta", "PRETA"), ("Parda", "PARDA"), ("Branca", "BRANCA")]:
            r = sb[(sb.rotulo_treino == tr) & (sb.rotulo_avaliacao == "Uniao") & (sb.grupo == g)].iloc[0]
            tag(f"C-UND-{tr}-{p}", -r.calibracao_na_media_pp, "{:.1f}", S, "calibracao_rotulos_subgrupos.csv",
                f"{tr}->Uniao.{g} (subestimação)")
    h = ler("calibracao_rotulos_hr.csv").set_index("covariavel")
    for cov, p in [("regiao_Sul", "SUL"), ("regiao_Sudeste", "SE")]:
        for rot in ("SIM", "RHC"):
            tag(f"C-HR-{p}-{rot}", h.loc[cov, f"HR_{rot}"], "{:.2f}", S, "calibracao_rotulos_hr.csv", f"{cov}.HR_{rot}")


def main() -> None:
    # ---------- D: Fellegi–Sunter ----------
    S = "scripts/fs_diagnostico_u_m.py"
    u = ler("fs_u_corrigido_coorte.csv").set_index("chave")
    k3 = "nasc+sexo+mun_res"
    tag("D-RATIO-OLD", u.loc[k3, "razao_com_autopar"], "{:.1f}", S, "fs_u_corrigido_coorte.csv", f"{k3}.razao_com_autopar")
    tag("D-RATIO-NOSELF", u.loc[k3, "razao_sem_autopar"], "{:.1f}", S, "fs_u_corrigido_coorte.csv", f"{k3}.razao_sem_autopar")
    tag("D-SELFSHARE", 100 * (1 - u.loc[k3, "u_sem_autopar"] / u.loc[k3, "u_com_autopar"]), "{:.0f}", S,
        "fs_u_corrigido_coorte.csv", f"{k3}: 1 − u_sem/u_com")
    tag("D-POOL", u.loc[k3, "N_pool"], "{:,.0f}", S, "fs_u_corrigido_coorte.csv", f"{k3}.N_pool")
    dup = ler("fs_duplicatas_resumo_coorte.csv").set_index("criterio")
    tag("D-PAIRS", dup.iloc[0]["n_pares"], "{:,.0f}", S, "fs_duplicatas_resumo_coorte.csv", "linha 1 n_pares")
    for rot, nome in [("mesmo CNES em algum momento", "D-DUP-CNES"), ("mesmo CNES no mesmo ano", "D-DUP-ANO"),
                      ("mesmo CNES na mesma competência (mês)", "D-DUP")]:
        tag(nome, dup.loc[rot, "n_pares"], "{:,.0f}", S, "fs_duplicatas_resumo_coorte.csv", f"{rot}.n_pares")
        tag(nome + "-PCT", dup.loc[rot, "pct"], "{:.1f}", S, "fs_duplicatas_resumo_coorte.csv", f"{rot}.pct")
    sd = ler("fs_u_sem_duplicatas_coorte.csv").set_index("duplicatas_removidas_por")
    tag("D-RATIO", sd.loc["mesmo_cnes_mes", "razao"], "{:.1f}", S, "fs_u_sem_duplicatas_coorte.csv", "mesmo_cnes_mes.razao")
    tag("D-RATIO-ALLCNES", sd.loc["mesmo_cnes", "razao"], "{:.1f}", S, "fs_u_sem_duplicatas_coorte.csv", "mesmo_cnes.razao")
    tag("D-U", sd.loc["mesmo_cnes_mes", "u_conjunto"], "{:.2e}", S, "fs_u_sem_duplicatas_coorte.csv", "mesmo_cnes_mes.u_conjunto")
    un = ler("fs_u_corrigido_nacional.csv").set_index("chave")
    tag("D-RATIO-NAT", un.loc[k3, "razao_sem_autopar"], "{:.2f}", S, "fs_u_corrigido_nacional.csv", f"{k3}.razao_sem_autopar")
    mn = ler("fs_m_conjunto_nacional.csv").set_index("chave")
    tag("D-M", mn.loc[k3, "m_medido"], "{:.3f}", S, "fs_m_conjunto_nacional.csv", f"{k3}.m_medido")
    tag("D-M-PROD", mn.loc[k3, "m_produto"], "{:.3f}", S, "fs_m_conjunto_nacional.csv", f"{k3}.m_produto")
    tag("D-BITS-NAT", mn.loc[k3, "peso_bits_corrigido"], "{:.2f}", S, "fs_m_conjunto_nacional.csv", f"{k3}.peso_bits_corrigido")
    mc = ler("fs_m_conjunto_coorte.csv").set_index("chave")
    tag("D-M-COH", mc.loc[k3, "m_medido"], "{:.3f}", S, "fs_m_conjunto_coorte.csv", f"{k3}.m_medido")

    # ---------- Cenários / RHC ----------
    S = "projetos/Nefroblastoma_sensibilidade/cenarios_degradacao_linkage.py"
    c = ler("tabela_cenarios_degradacao.csv").set_index("cenario")
    for cen, p in [("0_referencia", "S0"), ("1_claims_ingenuo", "S1"), ("1b_sem_link_SIM", "S1B"),
                   ("2_ambiguidade_removida", "S2")]:
        tag(f"{p}-DEATHS", c.loc[cen, "obitos"], "{:,.0f}", S, "tabela_cenarios_degradacao.csv", f"{cen}.obitos")
        tag(f"{p}-OS", c.loc[cen, "os_5a_pct"], "{:.1f}", S, "tabela_cenarios_degradacao.csv", f"{cen}.os_5a_pct")
        tag(f"{p}-HR", c.loc[cen, "hr_estadio_iv_vs_i"], "{:.2f}", S, "tabela_cenarios_degradacao.csv", f"{cen}.hr")
    S = "projetos/Nefroblastoma_sensibilidade/concordancia_rhc_sus.py"
    r = ler("tabela_rhc_x_sus.csv").set_index("horizonte").loc["completo"]
    tag("A-RHC", r.pct_rhc_achados_pelo_pipeline, "{:.1f}", S, "tabela_rhc_x_sus.csv", "completo.pct")
    tag("A-RHC-LO", r.ic_lo, "{:.1f}", S, "tabela_rhc_x_sus.csv", "completo.ic_lo")
    tag("A-RHC-HI", r.ic_hi, "{:.1f}", S, "tabela_rhc_x_sus.csv", "completo.ic_hi")
    tag("A-RHC-N", r.obitos_rhc, "{:,.0f}", S, "tabela_rhc_x_sus.csv", "completo.obitos_rhc")
    f = ler("tabela_obitos_c1_por_fonte.csv").query("horizonte == 'completo'").set_index("origem")
    tag("SIH-ONLY", f.loc["so_SIH", "n"], "{:,.0f}", S, "tabela_obitos_c1_por_fonte.csv", "so_SIH.n")
    tag("SIH-CONF", f.loc["so_SIH", "confirmados_rhc"], "{:,.0f}", S, "tabela_obitos_c1_por_fonte.csv", "so_SIH.confirmados")
    tag("SIH-CONF-PCT", f.loc["so_SIH", "pct_confirmados"], "{:.1f}", S, "tabela_obitos_c1_por_fonte.csv", "so_SIH.pct")
    s = ler("tabela_sensibilidade_sih.csv").set_index("cenario").loc["S2_censura_so_SIH_nao_confirmado"]
    tag("SIH-CENS-N", s.obitos_censurados, "{:,.0f}", S, "tabela_sensibilidade_sih.csv", "S2.obitos_censurados")
    tag("SIH-CENS-OS", s.os_5a_pct, "{:.1f}", S, "tabela_sensibilidade_sih.csv", "S2.os_5a_pct")
    tag("SIH-CENS-HR", s.hr_estadio_iv, "{:.2f}", S, "tabela_sensibilidade_sih.csv", "S2.hr_estadio_iv")

    # ---------- B: propagação ----------
    S = "projetos/Nefroblastoma_sensibilidade/propagacao_erro_linkage.py"
    cj = ler("propagacao_conjuntos.csv").set_index("conjunto")["n"]
    tag("B-DEP", cj.iloc[3], "{:,.0f}", S, "propagacao_conjuntos.csv", "linha 4")
    tag("B-DEP-SIM", cj.iloc[1], "{:,.0f}", S, "propagacao_conjuntos.csv", "linha 2")
    tag("B-DEP-SIH", cj.iloc[2], "{:,.0f}", S, "propagacao_conjuntos.csv", "linha 3")
    tx = ler("propagacao_taxas_medidas.csv").set_index("cenario")
    m = tx.loc["FNR medida por completude"]
    tag("B-OS", m.os5_med, "{:.1f}", S, "propagacao_taxas_medidas.csv", "FNR por completude.os5_med")
    tag("B-OS-LO", m.os5_p025, "{:.1f}", S, "propagacao_taxas_medidas.csv", "os5_p025")
    tag("B-OS-HI", m.os5_p975, "{:.1f}", S, "propagacao_taxas_medidas.csv", "os5_p975")
    tag("B-HR", m.hr_med, "{:.2f}", S, "propagacao_taxas_medidas.csv", "hr_med")
    tag("B-HR-LO", m.hr_p025, "{:.2f}", S, "propagacao_taxas_medidas.csv", "hr_p025")
    tag("B-HR-HI", m.hr_p975, "{:.2f}", S, "propagacao_taxas_medidas.csv", "hr_p975")
    fp = tx.loc["FNR medida por completude + vínculos falsos, todo ambíguo C2 (66/314)"]
    tag("B-FP-OS", fp.os5_med, "{:.1f}", S, "propagacao_taxas_medidas.csv", "+falsos 21%.os5_med")
    tag("B-FP-HR", fp.hr_med, "{:.2f}", S, "propagacao_taxas_medidas.csv", "+falsos 21%.hr_med")
    lim = ler("propagacao_limiares.csv").set_index("deslocamento_os_pp")
    tag("B-1PT", 100 * lim.loc[1.0, "perda_necessaria"], "{:.0f}", S, "propagacao_limiares.csv", "1.0.perda_necessaria")
    tag("B-MAX", lim.loc[1.0, "deslocamento_maximo_possivel_pp"], "{:.1f}", S, "propagacao_limiares.csv", "deslocamento_maximo")
    g = ler("propagacao_gap_racial.csv")
    tag("B-GAP-REF", g.iloc[0].gap_med, "{:.1f}", S, "propagacao_gap_racial.csv", "fnr_branca=0.gap_med")
    gm = g.iloc[(g.fnr_branca - 0.025).abs().argmin()]
    tag("B-GAP", gm.gap_med, "{:.1f}", S, "propagacao_gap_racial.csv", "fnr_branca=0.025.gap_med")
    gx = g.iloc[-1]
    tag("B-GAP-30", gx.gap_med, "{:.1f}", S, "propagacao_gap_racial.csv", "fnr_branca=0.30.gap_med")

    # ---------- F: Tabela 2 (raça faltante) ----------
    S = "projetos/Nefroblastoma_sensibilidade/tabela2_raca_faltante.py"
    t2 = ler("tabela2_revisada.csv").set_index("metodo")
    for met, p in [("Complete case", "F-CC"), ("IPW, P(raça observada)", "F-IPW"), ("Imputação múltipla", "F-MI")]:
        r = t2.loc[met]
        NUM[p] = {"valor": f"{r.hr_white_vs_black:.2f} [{r.hr_ic_lo:.2f}, {r.hr_ic_hi:.2f}]",
                  "script": S, "arquivo": "outputs/tabela2_revisada.csv", "celula": f"{met}.hr (ic)"}
        tag(p + "-GAP", r.gap_pp, "{:.1f}", S, "tabela2_revisada.csv", f"{met}.gap_pp")
    tag("F-IMP", t2.loc["Imputação múltipla", "n_imputacoes"], "{:.0f}", S, "tabela2_revisada.csv", "MI.n_imputacoes")
    w = ler("tabela2_pesos_ipw.csv").iloc[0]
    tag("F-W-MIN", w["min"], "{:.2f}", S, "tabela2_pesos_ipw.csv", "min")
    tag("F-W-MAX", w["max"], "{:.2f}", S, "tabela2_pesos_ipw.csv", "max")

    # ---------- F: entropia, decomposição, u intra-UF; E-YEAR: padronização por ano ----------
    mm = ler("entropia_miller_madow_ate2025.csv").set_index("faixa").loc["total"]
    S = "scripts/entropia_miller_madow_decomposicao.py"
    tag("F-H-PLUGIN", mm.entropia_plugin_media, "{:.3f}", S, "entropia_miller_madow_ate2025.csv", "total.plugin")
    tag("F-H-MM", mm.entropia_mm_media, "{:.3f}", S, "entropia_miller_madow_ate2025.csv", "total.mm")
    if (O / "decomposicao_identidades_ate2025.csv").exists():
        dc = ler("decomposicao_identidades_ate2025.csv").set_index("categoria")
        tag("F-MUN", dc.loc["RESUMO: só município varia (mudança de endereço plausível)", "pct_dos_multi"],
            "{:.1f}", S, "decomposicao_identidades_ate2025.csv", "RESUMO só município.pct")
        tag("F-DOB", dc.loc["RESUMO: nascimento ou sexo varia (contaminação do identificador)", "pct_dos_multi"],
            "{:.1f}", S, "decomposicao_identidades_ate2025.csv", "RESUMO nasc ou sexo.pct")
    S = "scripts/score_pares_rescore_intra_uf.py"
    sc = ler("distribuicao_score_intraUF_ate2025.csv")
    NUM["F-ORDER"] = {"valor": "monotone" if sc.ordem_nova_monotona.all() else "NOT monotone", "script": S,
                      "arquivo": "outputs/distribuicao_score_intraUF_ate2025.csv", "celula": "ordem_nova_monotona"}
    uu = ler("u_m_empirico_por_uf_ate2025.csv")
    tag("F-UMUN-UF", (uu.u_mun * uu.n_pares_falsos).sum() / uu.n_pares_falsos.sum(), "{:.3f}", S,
        "u_m_empirico_por_uf_ate2025.csv", "u_mun ponderado")
    S = "scripts/tabela1_cns_padronizada_ano.py"
    y = ler("tabela1_raca_padronizada_ano_ate2025.csv").set_index("raca_cor")
    for raca, p in [("Preta", "E-YEAR-PRETA"), ("Branca", "E-YEAR-BRANCA"), ("Parda", "E-YEAR-PARDA"),
                    ("Sem informação", "E-YEAR-SI")]:
        tag(p, y.loc[raca, "pct_nao_confiavel_padronizado_ano"], "{:.1f}", S,
            "tabela1_raca_padronizada_ano_ate2025.csv", f"{raca}.padronizado")
        tag(p + "-RAW", y.loc[raca, "pct_nao_confiavel_bruto"], "{:.1f}", S,
            "tabela1_raca_padronizada_ano_ate2025.csv", f"{raca}.bruto")

    # ---------- E: raça faltante ajustada; F: IC por cluster e trincas (se o CSV existir) ----------
    if (O / "efeito_raca_faltante_ate2025.csv").exists():
        S = "scripts/completude_modelo_ajustado.py"
        e = ler("efeito_raca_faltante_ate2025.csv").set_index("estimativa")
        tag("E-RAW", e.iloc[0].rd_pontos, "{:.1f}", S, "efeito_raca_faltante_ate2025.csv", "bruto.rd")
        for i, p in [(1, "E-ADJ"), (2, "E-FAC")]:
            r = e.iloc[i]
            NUM[p] = {"valor": f"{r.rd_pontos:.1f} [{r.ic_lo:.1f}, {r.ic_hi:.1f}]", "script": S,
                      "arquivo": "outputs/efeito_raca_faltante_ate2025.csv", "celula": f"{e.index[i]}.rd (ic)"}
        tag("E-BLQ", e.iloc[3].rd_pontos, "{:.1f}", S, "efeito_raca_faltante_ate2025.csv", "bloqueáveis.rd")
    if (O / "paridade_fnr_cluster_ate2025.csv").exists():
        S = "scripts/completude_modelo_ajustado.py"
        cl = ler("paridade_fnr_cluster_ate2025.csv")
        for comp, p in [("Preto/Pardo vs Branco", "F-CI1"), ("Norte/Nordeste vs Sul/Sudeste", "F-CI2")]:
            r = cl[(cl.comparacao == comp) & (cl.unidade_cluster == "mun_res")].iloc[0]
            NUM[p] = {"valor": f"{r.diferenca_pp:.2f} [{r.ic_lo_pp:.2f}, {r.ic_hi_pp:.2f}]", "script": S,
                      "arquivo": "outputs/paridade_fnr_cluster_ate2025.csv", "celula": f"{comp}.mun_res"}
        for comp, p in [("Preto/Pardo vs Branco", "F-CI1-UF"), ("Norte/Nordeste vs Sul/Sudeste", "F-CI2-UF")]:
            r = cl[(cl.comparacao == comp) & (cl.unidade_cluster == "uf")].iloc[0]
            NUM[p] = {"valor": f"{r.diferenca_pp:.2f} [{r.ic_lo_pp:.2f}, {r.ic_hi_pp:.2f}]", "script": S,
                      "arquivo": "outputs/paridade_fnr_cluster_ate2025.csv", "celula": f"{comp}.uf"}
        NUM["F-UNIT"] = {"valor": "municipality of residence", "script": S,
                         "arquivo": "outputs/paridade_fnr_cluster_ate2025.csv", "celula": "unidade_cluster"}
    if (O / "colisao_trinca_baseline_ate2025.csv").exists():
        S = "scripts/colisao_trinca_baseline_acaso.py"
        b = ler("colisao_trinca_baseline_ate2025.csv").iloc[0]
        tag("F-CHANCE", b.pct_multi_esperado_acaso, "{:.1f}", S, "colisao_trinca_baseline_ate2025.csv", "pct_esperado")
        tag("F-CHANCE-RATIO", b.razao_obs_esperado, "{:.2f}", S, "colisao_trinca_baseline_ate2025.csv", "razao")

    # ---------- A: cascata ----------
    S = "scripts/cascade_bi_recall_interno.py"
    rc = ler("recall_chaves_ate2025.csv").set_index("chave")
    for ch, p in [("R1 (nasc+CNES)", "A-R1"), ("R3 (nasc+sexo+mun_res)", "A-R3"), ("R4 (nasc+sexo+mun_ubs)", "A-R4")]:
        tag(p, 100 * rc.loc[ch, "taxa_recall"], "{:.1f}", S, "recall_chaves_ate2025.csv", f"{ch}.taxa_recall")
    if (O / "cascata_registro_retido_ate2025.csv").exists():
        S = "scripts/cascata_registro_retido_uf.py"
        ca = ler("cascata_registro_retido_ate2025.csv")
        ca = ca[(ca.uf == "todas") & (ca.regra == "cascata (acumulada)") & (ca.ordem == "produção (R1→R3→R4)")]
        uf_ = ca[ca.pool == "UF inteira"].iloc[0]
        coh = ca[ca.pool != "UF inteira"].iloc[0]
        # principal = pool do tamanho de coorte (o uso real: o pipeline filtra o CID dos dois lados)
        tag("A-REC", 100 * coh.recall, "{:.1f}", S, "cascata_registro_retido_ate2025.csv", "todas/coorte/cascata.recall")
        tag("A-PREC", 100 * coh.precisao, "{:.1f}", S, "cascata_registro_retido_ate2025.csv", "todas/coorte/cascata.precisao")
        tag("A-N", coh.consultas, "{:,.0f}", S, "cascata_registro_retido_ate2025.csv", "todas/coorte/cascata.consultas")
        tag("A-AMB", 100 * coh.ambiguas / coh.vinculadas, "{:.0f}", S, "cascata_registro_retido_ate2025.csv", "todas/coorte ambiguas/vinculadas")
        tag("A-REC-UF", 100 * uf_.recall, "{:.1f}", S, "cascata_registro_retido_ate2025.csv", "todas/UF/cascata.recall")
        tag("A-PREC-UF", 100 * uf_.precisao, "{:.1f}", S, "cascata_registro_retido_ate2025.csv", "todas/UF/cascata.precisao")
        tag("A-AMB-UF", 100 * uf_.ambiguas / uf_.vinculadas, "{:.0f}", S, "cascata_registro_retido_ate2025.csv", "todas/UF ambiguas/vinculadas")
        tag("A-N-UF", uf_.consultas, "{:,.0f}", S, "cascata_registro_retido_ate2025.csv", "todas/UF/cascata.consultas")

    # ---------- C: Cox C50 sob três rótulos ----------
    S = "scripts/c50_calibracao_rotulos.py"
    h = ler("calibracao_rotulos_hr.csv").set_index("covariavel")
    for cov, p in [("staging_IV", "C-HR-IV"), ("raca_label_Parda", "C-HR-PARDA"), ("raca_label_Preta", "C-HR-PRETA")]:
        for rot in ("SIM", "RHC", "Uniao"):
            tag(f"{p}-{rot}", h.loc[cov, f"HR_{rot}"], "{:.2f}", S, "calibracao_rotulos_hr.csv", f"{cov}.HR_{rot}")
    gl = ler("calibracao_rotulos_global.csv").set_index(["rotulo_treino", "rotulo_avaliacao"])
    for tr in ("SIM", "RHC", "Uniao"):
        for av in ("SIM", "RHC", "Uniao"):
            r = gl.loc[(tr, av)]
            tag(f"C-OE-{tr}-{av}", r.razao_previsto_observado, "{:.2f}", S, "calibracao_rotulos_global.csv",
                f"{tr}->{av}.razao")
            tag(f"C-C-{tr}-{av}", r.c_harrell, "{:.3f}", S, "calibracao_rotulos_global.csv", f"{tr}->{av}.c")
    tag("C-PRED-SIM", gl.loc[("SIM", "Uniao"), "risco_5a_previsto_medio"], "{:.1f}", S, "calibracao_rotulos_global.csv", "SIM->Uniao.previsto")
    tag("C-OBS-UNION", gl.loc[("SIM", "Uniao"), "risco_5a_observado_km"], "{:.1f}", S, "calibracao_rotulos_global.csv", "SIM->Uniao.observado")
    sb = ler("calibracao_rotulos_subgrupos.csv")
    r = sb[(sb.rotulo_treino == "RHC") & (sb.rotulo_avaliacao == "Uniao") & (sb.grupo == "Preta")].iloc[0]
    tag("C-SUB", r.calibracao_na_media_pp, "{:.1f}", S, "calibracao_rotulos_subgrupos.csv", "RHC->Uniao.Preta")
    tag("C-SUB-PRED", r.risco_5a_previsto_medio, "{:.1f}", S, "calibracao_rotulos_subgrupos.csv", "previsto")
    tag("C-SUB-OBS", r.risco_5a_observado_km, "{:.1f}", S, "calibracao_rotulos_subgrupos.csv", "observado")
    pop = ler("calibracao_rotulos_populacao.csv").set_index("passo")["n"]
    tag("C-N", pop.loc["covariáveis completas"], "{:,.0f}", S, "calibracao_rotulos_populacao.csv", "covariáveis completas")
    tag("C-N-LINKED", pop.loc["ligados RHC × coorte de sobrevida"], "{:,.0f}", S, "calibracao_rotulos_populacao.csv",
        "ligados RHC × coorte de sobrevida")

    extras()
    extras2()
    extras3()
    extras4()
    saida = AQUI.parent / "numbers" / "numbers_all.json"
    saida.write_text(json.dumps(NUM, indent=1, ensure_ascii=False))
    print(f"✅ {len(NUM)} números em {saida.name}")
    for k, v in NUM.items():
        print(f"  {k:22s} {v['valor']}")


if __name__ == "__main__":
    main()
