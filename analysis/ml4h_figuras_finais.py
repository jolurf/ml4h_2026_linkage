"""Figuras da versão final do manuscrito ML4H (`paper/ml4h_final.tex`).

Cada figura é desenhada no TAMANHO EXATO em que entra no PDF e incluída a
`width=\\linewidth`, sem reescala: página = 433,62 pt (6,0 pol.), coluna =
211,81 pt (2,93 pol.), medidas na classe jmlr. A fonte mínima é 9 pt na página
impressa.

Paleta (distinção conferida entre todos os pares, fundo claro):
azul = rótulo pipeline / vínculo correto / perder vínculos; laranja = registro de
câncer / vínculo errado / omitir uma fonte; água = união / óbitos não registrados;
verde (linha cheia) = validado com identificador; violeta (tracejado) = indireto.

Exceto a KM (precisa da coorte), tudo sai dos CSVs agregados de
`outputs/`, os mesmos de onde saem os números do texto.

    python analysis/ml4h_figuras_finais.py --sem-km   # só dos CSVs de outputs/
    python analysis/ml4h_figuras_finais.py            # todas; a KM exige o pipeline dos autores

Saídas em paper/figures/ (corpo do texto):
    fig1_overview.pdf          o problema e o desenho (Medir → Localizar → Propagar)
    fig2_sources_links.pdf     fontes, links e onde cada link tem gabarito
    fig3_cascade_pools.pdf     cascata com CNS escondido: correto / errado / sem vínculo, por pool
    fig4_race_recall.pdf       recall por raça e efeito da raça faltante (ajustado)
    fig5_death_budget.pdf      os óbitos da coorte por fonte e o link de que dependem
    fig6_scenarios.pdf         OS em 5 anos e HR do estádio IV em cada cenário de erro
    fig7_model_labels.pdf      Cox do C50 sob três rótulos: HR e calibração por raça
Apêndice:
    figB_propagation.pdf       curva de perda de vínculos e gap racial
    figC_km_scenarios.pdf      KM dos cenários de omissão de fonte
    figD_calibration.pdf       calibração na média, por rótulo de treino × avaliação
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]  # raiz deste repositório
OUT = REPO / "outputs"
FIG = REPO / "paper" / "figures"
try:  # dentro do pipeline dos autores: a KM (figC) lê a coorte
    import _raiz  # noqa: F401,E402
    import paths as _paths  # noqa: E402
    RAIZ = _paths.STATS.parents[1]
except ImportError:  # fora dele, só os CSVs agregados (use --sem-km)
    RAIZ = None

TW = 433.62 / 72.27   # \textwidth  (figure*)
CW = 211.81 / 72.27   # \columnwidth (figure)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
VERDE, VIOLETA = "#008300", "#6250d6"
CINZA_CLARO, FUNDO = "#c9c8c3", "#f3f2ee"
COR_ROTULO = {"SIM": AZUL, "RHC": LARANJA, "Uniao": AGUA}
NOME_ROTULO = {"SIM": "pipeline", "RHC": "registry", "Uniao": "union"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlelocation": "left", "pdf.fonttype": 42,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    "svg.hashsalt": "ml4h", "path.simplify": False,
})


def salvar(fig, nome: str) -> None:
    """Sem bbox_inches='tight': o tamanho físico do arquivo é o figsize, e o
    LaTeX inclui a 100%."""
    FIG.mkdir(exist_ok=True)
    meta = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
    fig.savefig(FIG / nome, metadata=meta)
    fig.savefig((FIG / nome).with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"🖼  {FIG / nome}")


def tela(w: float, h: float):
    """Eixo cobrindo a figura inteira, em polegadas — para esquemas."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def grade_x(ax) -> None:
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def fig1_overview() -> None:
    """(a) os registros de um paciente em quatro sistemas, só um com o
    identificador; (b) as três etapas do estudo, com o experimento e a métrica."""
    H = 2.78
    fig, ax = tela(TW, H)
    ax.text(0.02, H - 0.06, "(a) One patient, four national systems, one identifier", va="top", fontsize=9.5,
            fontweight="bold")
    linhas = [("Outpatient (SIA)", "health-card no."), ("Hospital claims (SIH)", "no identifier"),
              ("Death registry (SIM)", "no identifier"), ("Cancer registry (RHC)", "no identifier")]
    y0, dy, x0, x1 = H - 0.43, 0.21, 1.66, 4.75
    ys = [y0 - i * dy for i in range(4)]
    for (nome, ident), y in zip(linhas, ys):
        ax.text(0.02, y, nome, va="center", fontsize=9)
        ax.plot([x0, x1], [y, y], color=GRID, lw=0.8, zorder=1)
        com_id = ident.startswith("health")
        ax.text(x1 + 0.1, y, ident, va="center", fontsize=9, color=INK if com_id else INK2,
                fontweight="bold" if com_id else "normal")
    eventos = {0: [1.85, 2.3, 2.95, 3.45, 3.95, 4.4], 1: [2.65, 3.7], 2: [4.62], 3: [2.08]}
    for linha, xs in eventos.items():
        for x in xs:
            if linha == 0:
                ax.plot(x, ys[0], "o", ms=6.5, color=AZUL, mec="white", mew=0.8, zorder=3)
            else:
                ax.plot(x, ys[linha], "o", ms=6.5, mfc="white", mec=AZUL, mew=1.3, zorder=3)
                alvo = min(eventos[0], key=lambda e: abs(e - x))
                ax.plot([x, alvo], [ys[linha] + 0.06, ys[0] - 0.06], color=VIOLETA, lw=0.9, ls=(0, (2.5, 1.6)),
                        zorder=2)
    yt = ys[-1] - 0.2
    ax.annotate("", (x1, yt), (x0, yt), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
    ax.text(x1 + 0.1, yt, "time", va="center", fontsize=9, color=INK2)
    ax.plot([x0, x0 + 0.3], [yt - 0.2, yt - 0.2], color=VIOLETA, lw=0.9, ls=(0, (2.5, 1.6)))
    ax.text(x0 + 0.38, yt - 0.2, "attached by birth date or age, sex and place (the links audited here)", va="center",
            fontsize=9, color=INK2)

    ytit = 1.24
    ax.text(0.02, ytit, "(b) What we did", va="top", fontsize=9.5, fontweight="bold")
    cartoes = [("1  Measure", "Q1", "Hide the health-card\nnumber, re-link, and\nscore against it",
                "E1–E2 · recall, precision"),
               ("2  Locate", "Q2", "Who loses links? Recall\nby race and complete-\nness, adjusted",
                "E3 · FNR gap"),
               ("3  Propagate", "Q3", "Feed the measured\nerrors into survival and\na trained model",
                "E4–E5 · survival, HR"),
               ]
    w, gap, yb = 1.86, 0.21, 0.02
    h = ytit - 0.24 - yb
    for i, (tit, q, corpo, rodape) in enumerate(cartoes):
        x = 0.02 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, yb), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=FUNDO, edgecolor="none"))
        ax.text(x + 0.1, yb + h - 0.08, tit, va="top", fontsize=9.5, fontweight="bold")
        ax.text(x + w - 0.1, yb + h - 0.08, q, va="top", ha="right", fontsize=9, color=INK2)
        ax.text(x + 0.1, yb + h - 0.3, corpo, va="top", fontsize=9, linespacing=1.2)
        ax.text(x + 0.1, yb + 0.07, rodape, va="bottom", fontsize=9, color=INK2)
        if i < 2:
            ax.annotate("", (x + w + gap - 0.02, yb + h / 2), (x + w + 0.02, yb + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, mutation_scale=9))
    salvar(fig, "fig1_overview.pdf")


# ---------------------------------------------------------------------------
def fig2_sources_links() -> None:
    """Fontes e links. Linha cheia verde = há identificador dos dois lados
    (validação direta); tracejado violeta = sem identificador comum. Acima de
    cada seta: o método do link; em itálico: como ele foi validado."""
    H = 2.3
    fig, ax = tela(TW, H)
    w, h = 1.76, 0.5

    def caixa(xc, yc, titulo, sub, com_id=False):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=FUNDO, edgecolor=VERDE if com_id else "none", lw=1.1))
        ax.text(xc, yc + 0.09, titulo, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(xc, yc - 0.12, sub, ha="center", va="center", fontsize=9, color=INK2)

    def seta(p0, p1, direta):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.3,
                                     color=VERDE if direta else VIOLETA, linestyle="-" if direta else (0, (3, 2))))

    def rotulo(x, y, metodo, validacao, cor):
        ax.text(x, y, validacao, ha="center", va="bottom", fontsize=9, color=cor, style="italic", linespacing=1.15)
        ax.text(x, y + 0.17 * (validacao.count("\n") + 1) + 0.02, metodo, ha="center", va="bottom", fontsize=9)

    yt, yb = 1.42, 0.3
    xa, xb, xc, xr = 0.9, 3.0, 5.1, 3.6
    caixa(xa, yt, "Outpatient records", "SIA · ID + birth date", com_id=True)
    caixa(xa, yb, "Oncology records", "SIA · ID + age", com_id=True)
    caixa(xb, yt, "Hospital claims", "SIH · no ID")
    caixa(xc, yt, "Death registry", "SIM · no ID")
    caixa(xr, yb, "Cancer registry", "RHC · no ID")

    seta((xa + w / 2 + 0.02, yt), (xb - w / 2 - 0.02, yt), False)
    rotulo((xa + xb) / 2 + 0.15, yt + h / 2 + 0.06, "cascade R1–R4", "re-run with\nID hidden (E2)", VIOLETA)
    seta((xb + w / 2 + 0.02, yt), (xc - w / 2 - 0.02, yt), False)
    rotulo((xb + xc) / 2, yt + h / 2 + 0.06, "death link C2", "capture–recapture,\ndirect search (E4)", VIOLETA)
    seta((xa, yt - h / 2 - 0.02), (xa, yb + h / 2 + 0.02), True)
    ax.text(xa + 0.1, (yt + yb) / 2 + 0.08, "blind linkage", ha="left", va="bottom", fontsize=9)
    ax.text(xa + 0.1, (yt + yb) / 2 + 0.05, "ID hidden (E2)", ha="left", va="top", fontsize=9, color=VERDE,
            style="italic")
    seta((xr - w / 2 - 0.02, yb), (xa + w / 2 + 0.02, yb), False)
    ax.text((xa + xr) / 2 + 0.02, yb + 0.06, "registry link", ha="center", va="bottom", fontsize=9)
    ax.text((xa + xr) / 2 + 0.02, yb - 0.05, "surrogate (E4)", ha="center", va="top", fontsize=9, color=VIOLETA,
            style="italic")
    leg = [Line2D([], [], color=VERDE, lw=1.3, label="ID on both sides:\nvalidated directly"),
           Line2D([], [], color=VIOLETA, lw=1.3, ls=(0, (3, 2)), label="no shared ID:\nvalidated indirectly")]
    ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, handlelength=2.0,
              borderaxespad=0.05, labelspacing=0.6)
    salvar(fig, "fig2_sources_links.pdf")


# ---------------------------------------------------------------------------
def fig3_cascade_pools() -> None:
    """Cascata R1→R3→R4 com o CNS escondido: cada consulta termina num vínculo
    correto (= recall), num vínculo errado ou sem vínculo. Precisão = correto /
    (correto + errado)."""
    c = pd.read_csv(OUT / "cascata_registro_retido_ate2025.csv")
    prod = "produção (R1→R3→R4)"
    pools = [("1% dos pacientes (tamanho de coorte)", "Disease-sized pool"),
             ("1% dos pacientes (tamanho de coorte), nascidos >= 2005 (pediátrico)", "Disease-sized, born ≥ 2005"),
             ("UF inteira", "Whole state")]
    fig, ax = plt.subplots(figsize=(CW, 2.05), layout="constrained")
    for i, (k, nome) in enumerate(pools):
        r = c[(c.uf == "todas") & (c.pool == k) & (c.ordem == prod) & (c.regra == "cascata (acumulada)")].iloc[0]
        cor = 100 * r.corretas / r.consultas
        err = 100 * (r.vinculadas - r.corretas) / r.consultas
        sem = 100 - cor - err
        y = -i
        ax.barh(y, cor, color=AZUL, height=0.42, edgecolor="white", lw=1)
        ax.barh(y, err, left=cor, color=LARANJA, height=0.42, edgecolor="white", lw=1)
        ax.barh(y, sem, left=cor + err, color=CINZA_CLARO, height=0.42, edgecolor="white", lw=1)
        ax.text(1, y + 0.27, f"{nome}: precision {100 * r.precisao:.1f}%", va="bottom", fontsize=9)
        ax.text(cor - 2, y, f"{cor:.1f}", ha="right", va="center", fontsize=9, color="white", fontweight="bold")
        if err > 20:
            ax.text(cor + err - 2, y, f"{err:.1f}", ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.45, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of held-out records (%)")
    leg = [Patch(color=AZUL, label="correct"), Patch(color=LARANJA, label="wrong"), Patch(color=CINZA_CLARO, label="none")]
    fig.legend(handles=leg, loc="outside upper center", ncol=3, frameon=False, handlelength=1.0, columnspacing=1.0)
    salvar(fig, "fig3_cascade_pools.pdf")


# ---------------------------------------------------------------------------
def fig4_race_recall() -> None:
    m = pd.read_csv(OUT / "metricas_por_estrato_ate2025.csv").set_index("estrato")
    e = pd.read_csv(OUT / "efeito_raca_faltante_ate2025.csv").set_index("estimativa")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 2.35), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    racas = [("Black", "Preta"), ("White", "Branca"), ("Mixed", "Parda"), ("Asian", "Amarela"),
             ("Indi-\ngenous", "Indígena"), ("Miss-\ning", "Sem informação")]
    vals = [100 * m.loc[f"Raça/cor: {p}", "recall"] for _, p in racas]
    los = [100 * (m.loc[f"Raça/cor: {p}", "recall"] - m.loc[f"Raça/cor: {p}", "recall_ic_lo"]) for _, p in racas]
    his = [100 * (m.loc[f"Raça/cor: {p}", "recall_ic_hi"] - m.loc[f"Raça/cor: {p}", "recall"]) for _, p in racas]
    a.bar(range(6), vals, yerr=[los, his], color=[AZUL] * 5 + [LARANJA], width=0.62, capsize=2,
          error_kw=dict(ecolor=INK2, lw=0.8))
    for i, (v, h) in enumerate(zip(vals, his)):
        a.text(i, v + h + 0.35, f"{v:.1f}", ha="center", fontsize=9)
    a.set_xticks(range(6), [r for r, _ in racas])
    a.set_ylim(80, 101.5)
    a.set_ylabel("Recall, ID hidden (%)")
    a.set_title("(a) Recall by race/colour")

    linhas = [("Crude", "bruto"), ("Sex and region\nrecorded", "só registros com sexo e região informados"),
              ("Adjusted for\nyear and state", "ajustado por ano e UF (registros bloqueáveis)"),
              ("Within facility\nand year", "dentro do estabelecimento e ano (MH por CNES×ano, bootstrap por CNES)")]
    y = list(range(len(linhas)))[::-1]
    for yi, (rot, k) in zip(y, linhas):
        r = e.loc[k]
        x = -r.rd_pontos
        if pd.notna(r.ic_lo):
            b.errorbar(x, yi, xerr=[[x - (-r.ic_hi)], [(-r.ic_lo) - x]], fmt="o", color=LARANJA, capsize=2.5, ms=5,
                       ecolor=INK2, elinewidth=0.9)
        else:
            b.plot(x, yi, "o", color=LARANJA, ms=5)
        b.text(x, yi + 0.2, f"{x:.1f}", ha="center", va="bottom", fontsize=9)
    b.set_yticks(y, [r for r, _ in linhas])
    b.set_ylim(-0.6, len(linhas) - 0.25)
    b.set_xlim(0, 14)
    grade_x(b)
    b.set_xlabel("Recall gap (points)")
    b.set_title("(b) Race missing: gap")
    salvar(fig, "fig4_race_recall.pdf")


# ---------------------------------------------------------------------------
def fig5_death_budget() -> None:
    """Os óbitos em 5 anos por fonte. Azul: também no registro de câncer.
    Laranja: não confirmados e apoiados numa cascata auditada (podem ser perdidos
    pelo linkage). Cinza: não confirmados, mas no registro do próprio paciente.
    Água: só no registro de câncer (apoiados no link do registro)."""
    ob = pd.read_csv(OUT / "tabela_obitos_c1_por_fonte.csv")
    ob = ob[ob.horizonte == "5 anos"].set_index("origem")
    cj = pd.read_csv(OUT / "propagacao_conjuntos.csv")["n"].tolist()
    rx = pd.read_csv(OUT / "tabela_rhc_x_sus.csv").set_index("horizonte").loc["5 anos"]
    n_sim, perde_sim, perde_sih = cj[4], cj[1], cj[2]
    linhas = [("Outpatient (ID)", ob.loc["so_SIA", "n"], ob.loc["so_SIA", "confirmados_rhc"], False),
              ("Outpatient + hosp.", ob.loc["ambos", "n"], ob.loc["ambos", "confirmados_rhc"], False),
              ("Hospital only", ob.loc["so_SIH", "n"], ob.loc["so_SIH", "n"] - perde_sih, True),
              ("SIM link", n_sim, n_sim - perde_sim, True),
              ("Cancer reg. only", rx.so_rhc, None, False)]
    fig, ax = plt.subplots(figsize=(CW, 2.55), layout="constrained")
    for i, (nome, n, conf, auditado) in enumerate(linhas):
        y = -i
        if conf is None:
            ax.barh(y, n, color=AGUA, height=0.62, edgecolor="white", lw=1)
        else:
            ax.barh(y, conf, color=AZUL, height=0.62, edgecolor="white", lw=1)
            ax.barh(y, n - conf, left=conf, color=LARANJA if auditado else CINZA_CLARO, height=0.62,
                    edgecolor="white", lw=1)
        ax.text(n + 2, y, f"{n:.0f}", va="center", fontsize=9)
    ax.set_yticks([-i for i in range(len(linhas))], [l[0] for l in linhas])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Deaths within five years")
    grade_x(ax)
    leg = [Patch(color=AZUL, label="also in cancer reg."), Patch(color=CINZA_CLARO, label="own records"),
           Patch(color=LARANJA, label=f"can be lost ({perde_sim + perde_sih:.0f})"), Patch(color=AGUA, label="cancer reg. only")]
    fig.legend(handles=leg, loc="outside upper center", ncol=2, frameon=False, handlelength=0.9, columnspacing=0.8,
               handletextpad=0.4)
    salvar(fig, "fig5_death_budget.pdf")


# ---------------------------------------------------------------------------
def fig6_scenarios() -> None:
    """OS em 5 anos e HR do estádio IV em cada cenário, agrupados pela fonte de
    erro. Faixa cinza = IC 95% amostral da referência."""
    cen = pd.read_csv(OUT / "tabela_cenarios_degradacao.csv").set_index("cenario")
    tx = pd.read_csv(OUT / "propagacao_taxas_medidas.csv").set_index("cenario")
    cr = pd.read_csv(OUT / "captura_recaptura_cenarios.csv").set_index("cenario")
    ref = cen.loc["0_referencia"]

    def t(k):
        return tx.loc[k, "os5_med"], tx.loc[k, "hr_med"]

    def c(k):
        return cr.loc[k, "os5_med"], cr.loc[k, "hr_med"]

    def s(k):
        return cen.loc[k, "os_5a_pct"], cen.loc[k, "hr_estadio_iv_vs_i"]

    casc = 100 * tx.loc["perda da cascata com CNS escondido", "taxa_uniforme"]
    impl = 100 * tx.loc["perda do pipeline implicada pela captura-recaptura", "taxa_uniforme"]
    k_falt = int(cr.loc["faltantes por estádio (principal)", "k_total"])
    k_fals = -int(cr.loc["só remover os prováveis falsos do RHC", "k_total"])
    linhas = [
        ("Reference: all sources linked", s("0_referencia"), INK),
        ("Losing existing links", None, AZUL),
        ("at the file-pair FNR", t("FNR medida por completude"), AZUL),
        (f"at the cascade's loss ({casc:.1f}%)", t("perda da cascata com CNS escondido"), AZUL),
        (f"at the implied loss ({impl:.1f}%)", t("perda do pipeline implicada pela captura-recaptura"), AZUL),
        ("+ false death links (upper bound)", t("FNR medida por completude + vínculos falsos, todo ambíguo C2 (66/314)"), AZUL),
        ("ambiguous-key deaths censored", s("2_ambiguidade_removida"), AZUL),
        ("Omitting a death source", None, LARANJA),
        ("no SIM link", s("1b_sem_link_SIM"), LARANJA),
        ("claims only (no SIM, no registry)", s("1_claims_ingenuo"), LARANJA),
        ("Unrecorded and false deaths", None, AGUA),
        (f"+ {k_falt} unrecorded deaths", c("faltantes por estádio (principal)"), AGUA),
        (f"− {k_fals} likely false RHC deaths", c("só remover os prováveis falsos do RHC"), AGUA),
        ("both corrections", c("remover os prováveis falsos do RHC e acrescentar os faltantes ajustados"), AGUA),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 3.35), layout="constrained", sharey=True,
                               gridspec_kw={"width_ratios": [1.25, 1]})
    ys = list(range(len(linhas)))[::-1]
    for ax, j, (lo, hi), lim, fmt in [(a, 0, (ref.os_5a_ic_lo, ref.os_5a_ic_hi), (76, 92), "{:.1f}"),
                                      (b, 1, (ref.hr_ic_lo, ref.hr_ic_hi), (1.8, 4.1), "{:.2f}")]:
        ax.axvspan(lo, hi, color=GRID, lw=0, zorder=0)
        ax.axvline(ref.os_5a_pct if j == 0 else ref.hr_estadio_iv_vs_i, color=INK2, lw=0.8, ls="--", zorder=1)
        for y, (rot, val, cor) in zip(ys, linhas):
            if val is None:
                continue
            ax.plot(val[j], y, "o", ms=6, color=cor, mec="white", mew=0.8, zorder=3)
            ax.annotate(fmt.format(val[j]), (val[j], y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9)
        ax.set_xlim(*lim)
        ax.tick_params(axis="y", length=0)
    a.set_yticks(ys, [("" if v is None else "   ") + r for r, v, _ in linhas])
    for tl, (_, v, _) in zip(a.get_yticklabels(), linhas):
        if v is None:
            tl.set_fontweight("bold")
    a.set_ylim(-0.6, len(linhas) - 0.3)
    a.set_xlabel("5-year overall survival (%)")
    b.set_xlabel("HR, stage IV vs. I")
    a.set_title("(a) Survival")
    b.set_title("(b) Stage IV HR")
    b.spines["left"].set_visible(False)
    salvar(fig, "fig6_scenarios.pdf")


# ---------------------------------------------------------------------------
def fig7_model_labels() -> None:
    """Cox do C50 treinado sob três rótulos de óbito: o que muda no coeficiente
    e na calibração por raça (contra a união, no teste)."""
    h = pd.read_csv(OUT / "calibracao_rotulos_hr.csv").set_index("covariavel")
    sb = pd.read_csv(OUT / "calibracao_rotulos_subgrupos.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TW, 2.15), layout="constrained",
                                  gridspec_kw={"width_ratios": [1, 1, 1.25]})
    rot = ["SIM", "RHC", "Uniao"]
    ys = [2, 1, 0]
    for ax, cov, tit, lim, ref in [(a, "staging_IV", "(a) Stage IV vs. I", (0, 9.5), None),
                                   (b, "raca_label_Parda", "(b) Mixed vs. White", (0.55, 1.25), 1.0)]:
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls="--")
        for y, r in zip(ys, rot):
            v, lo, hi = h.loc[cov, f"HR_{r}"], h.loc[cov, f"ic95_inf_{r}"], h.loc[cov, f"ic95_sup_{r}"]
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=COR_ROTULO[r], ms=6, capsize=2,
                        ecolor=INK2, elinewidth=0.9)
            ax.text(v, y + 0.22, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_yticks(ys, [f"{NOME_ROTULO[r]} label" for r in rot] if ax is a else [])
        ax.set_ylim(-0.5, 2.7)
        ax.set_xlim(*lim)
        ax.set_xlabel("Hazard ratio")
        if ax is a:
            ax.set_xticks([0, 2, 4, 6, 8])
        ax.tick_params(axis="y", length=0)
        grade_x(ax)
        ax.set_title(tit)
    b.spines["left"].set_visible(False)
    s = sb[(sb.rotulo_avaliacao == "Uniao") & (sb.eixo == "raca_label")]
    grupos = [("Branca", "White"), ("Parda", "Mixed"), ("Preta", "Black")]
    yg = [2, 1, 0]
    for k, (tr, dy) in enumerate([("SIM", 0.17), ("RHC", -0.17)]):
        ss = s[s.rotulo_treino == tr].set_index("grupo")
        for y, (g, _) in zip(yg, grupos):
            v = -ss.loc[g, "calibracao_na_media_pp"]
            c.plot([0, v], [y + dy, y + dy], color=COR_ROTULO[tr], lw=2.2, solid_capstyle="round")
            c.text(v + 0.3, y + dy, f"{v:.1f}", va="center", fontsize=9)
    c.set_yticks(yg, [n for _, n in grupos])
    c.set_xlim(0, 18)
    c.set_ylim(-0.5, 2.7)
    c.tick_params(axis="y", length=0)
    grade_x(c)
    c.set_xlabel("Under-prediction (points)")
    c.set_title("(c) Calibration by race")
    fig.get_layout_engine().set(wspace=0.06)
    leg = [Line2D([], [], color=COR_ROTULO[r], marker="o", lw=0, ms=6, label=f"trained on {NOME_ROTULO[r]} label")
           for r in rot]
    fig.legend(handles=leg, loc="outside lower center", ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.4)
    salvar(fig, "fig7_model_labels.pdf")


# ---------------------------------------------------------------------------
def figB_propagation() -> None:
    v = pd.read_csv(OUT / "propagacao_varredura.csv")
    g = pd.read_csv(OUT / "propagacao_gap_racial.csv")
    tx = pd.read_csv(OUT / "propagacao_taxas_medidas.csv").set_index("cenario")
    cr = pd.read_csv(OUT / "captura_recaptura_cenarios.csv").set_index("cenario")
    cen = pd.read_csv(OUT / "tabela_cenarios_degradacao.csv").set_index("cenario")
    dep = pd.read_csv(OUT / "propagacao_conjuntos.csv")["n"].tolist()[3]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    s = v[v.alvo == "S_SIM ∪ S_SIH"]
    lo, hi = cr.loc["faltantes por estádio (principal)", "os5_med"], cr.loc["só remover os prováveis falsos do RHC", "os5_med"]
    a.axhspan(lo, hi, color=AGUA, alpha=.15, lw=0)
    a.fill_between(100 * s.p_perda, s.os5_p025, s.os5_p975, color=AZUL, alpha=.25, lw=0)
    a.plot(100 * s.p_perda, s.os5_med, color=AZUL, lw=1.6, label=f"losing the {dep:.0f} cascade-dependent links")
    ref, s1 = cen.loc["0_referencia", "os_5a_pct"], cen.loc["1_claims_ingenuo", "os_5a_pct"]
    a.axhline(ref, color=INK2, lw=0.8, ls="--")
    a.text(99, ref - 0.2, f"reference {ref:.1f}", fontsize=9, va="top", ha="right")
    a.text(99, s1 - 0.2, f"claims only {s1:.1f}", fontsize=9, va="top", ha="right", color=INK2)
    a.axhline(s1, color=INK2, lw=0.8, ls=":")
    a.text(1, lo + 0.15, f"unrecorded / false deaths: {lo:.1f}–{hi:.1f}", fontsize=9, va="bottom")
    fnr_nac = pd.read_csv(OUT / "metricas_por_estrato_ate2025.csv").set_index("estrato").loc["Nacional", "fnr"]
    for k, rot_, off in [("FNR medida por completude", "file-pair FNR", (-2, -13)), ("perda da cascata com CNS escondido", "cascade", (5, 4)),
                         ("perda do pipeline implicada pela captura-recaptura", "implied", (5, -12))]:
        r = tx.loc[k]
        x = 100 * (r.taxa_uniforme if pd.notna(r.get("taxa_uniforme", float("nan"))) else fnr_nac)
        a.plot(x, r.os5_med, "o", color=AZUL, mec="white", ms=6, zorder=5)
        a.annotate(rot_, (x, r.os5_med), xytext=off, textcoords="offset points", fontsize=9)
    a.set_xlim(0, 100)
    a.set_ylim(76.5, 90.2)
    a.set_xlabel("Share of cascade-dependent death links lost (%)")
    a.set_ylabel("5-year overall survival (%)")
    a.set_title("(a) Losing links")
    b.fill_between(100 * g.fnr_branca, g.gap_p025, g.gap_p975, color=AZUL, alpha=.25, lw=0)
    b.plot(100 * g.fnr_branca, g.gap_med, color=AZUL, lw=1.6)
    b.axhline(g.iloc[0].gap_med, color=INK2, lw=0.8, ls="--")
    b.set_xlabel("False-negative rate, White (%)")
    b.set_ylabel("Survival gap (points)")
    b.set_title("(b) White − Black/mixed gap")
    salvar(fig, "figB_propagation.pdf")


def figC_km_scenarios() -> None:
    nef = RAIZ / "projetos" / "Nefroblastoma_sensibilidade"
    sys.path.insert(0, str(nef))
    from lifelines import KaplanMeierFitter
    import cenarios_degradacao_linkage as C
    _, cen = C.construir_tabela()
    rot = {"0_referencia": ("Reference", INK, "-"),
           "1b_sem_link_SIM": ("No SIM link", LARANJA, "--"),
           "1_claims_ingenuo": ("Claims only", LARANJA, "-")}
    fig, ax = plt.subplots(figsize=(CW, 2.4), layout="constrained")
    for k, (r, cor, ls) in rot.items():
        c = cen[k]
        KaplanMeierFitter().fit(c["T_meses"], c["E_final"], label=r).plot_survival_function(
            ax=ax, ci_show=False, color=cor, ls=ls, lw=1.4)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Overall survival")
    ax.legend(loc="lower left", frameon=False, handlelength=1.8)
    salvar(fig, "figC_km_scenarios.pdf")


def figD_calibration() -> None:
    gl = pd.read_csv(OUT / "calibracao_rotulos_global.csv")
    fig, ax = plt.subplots(figsize=(CW, 2.3), layout="constrained")
    rot = ["SIM", "RHC", "Uniao"]
    for i, tr in enumerate(rot):
        for j, av in enumerate(rot):
            r = gl[(gl.rotulo_treino == tr) & (gl.rotulo_avaliacao == av)].iloc[0]
            ax.bar(i * 4 + j, r.risco_5a_previsto_medio - r.risco_5a_observado_km, color=COR_ROTULO[av], width=0.85)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([1, 5, 9], [f"trained on\n{NOME_ROTULO[t]}" for t in rot])
    ax.set_ylabel("Predicted − observed (points)")
    leg = [Patch(color=COR_ROTULO[av], label=NOME_ROTULO[av]) for av in rot]
    fig.legend(handles=leg, loc="outside upper center", ncol=3, frameon=False, title="evaluated on", handlelength=1.0,
               title_fontsize=9)
    salvar(fig, "figD_calibration.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-km", action="store_true")
    a = ap.parse_args()
    fig1_overview()
    fig2_sources_links()
    fig3_cascade_pools()
    fig4_race_recall()
    fig5_death_budget()
    fig6_scenarios()
    fig7_model_labels()
    figB_propagation()
    figD_calibration()
    if not a.sem_km:
        if RAIZ is None:
            sys.exit("figC (KM) lê a coorte do pipeline dos autores; fora dele use --sem-km")
        figC_km_scenarios()


if __name__ == "__main__":
    main()
