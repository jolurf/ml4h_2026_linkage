# Auditing Identity Noise and Linkage Degradation at National Scale

Code and aggregate outputs for the ML4H 2026 (Findings Track) paper *Auditing
Identity Noise and Linkage Degradation at National Scale: From Bias Measurement to
Downstream Clinical Distortion*.

João Luis Reis Freitas¹, Leonardo Dexheimer da Silva¹, Felipe Matias Berg²³,
Nelson Wolosker¹²

¹ Faculdade de Medicina, Universidade de São Paulo, São Paulo, Brazil ·
² Hospital Israelita Albert Einstein, São Paulo, Brazil ·
³ Department of Radiology, Case Western Reserve University, Cleveland, OH, USA

Machine learning on health records assumes that each patient's history has been
stitched together correctly. In Brazil's public health system (SUS) only
outpatient records carry the health-card number (CNS); hospital claims, death
certificates and the hospital cancer registry are attached to a patient by birth
date, sex and place. We audit that linkage across 3.9 billion outpatient records
(2008–2025) and follow the measured errors into a clinical estimate and a trained
model. With the CNS hidden and used only as ground truth, the linkage cascade
reaches 94.6% recall and 96.0% precision in a candidate pool the size of a disease
cohort, but about 37% recall and precision across a whole state. Recall is 12.0 points lower where
race is not recorded (6.0 after adjusting for year and state). In 2,684 children
with nephroblastoma, losing links at the measured rates moves 5-year survival from
80.5% to at most 81.7%, while omitting the mortality sources moves it to 89.1%. In
a breast-cancer survival model (209,387 patients), the source that labels death
moves the stage IV hazard ratio from 2.52 to 8.07 and changes calibration by race.

Every number in the paper is read from an aggregate table in `outputs/` by
`analysis/coletar_numeros.py`; `numbers/provenance.md` maps each number to its
script, table and cell. No record-level data are included.

## Data sources

All inputs are open microdata published by the Brazilian government and
downloadable without registration:

- **DATASUS** (outpatient SIA: BI, AQ, AR; inpatient SIH; mortality SIM):
  <https://datasus.saude.gov.br/transferencia-de-arquivos/>
- **INCA IntegradorRHC** (hospital cancer registry, RHC; choose the case base and
  the years to obtain the yearly record-level files):
  <https://irhc.inca.gov.br/RHCNet/visualizaTabNetExterno.action>

## What is here

```
paper/       ml4h_final.tex.in (source with {{TAG}} placeholders), gerar_tex_final.py,
             ml4h_final.tex (generated), refs.bib, figures/ (the 10 figures, .pdf and .png),
             jmlr.cls and jmlrutils.sty (the official ML4H 2026 template class)
numbers/     numbers.json (tag -> value, script, output file, cell; the 271 tags the paper uses),
             provenance.md (the same as a table), make_provenance.py
outputs/     42 aggregate tables (counts, rates, estimates and CIs by stratum or scenario)
analysis/    the 21 scripts that produced outputs/, the figure script ml4h_figuras_finais.py,
             and coletar_numeros.py (outputs/ -> numbers)
synthetic/   two self-contained scripts on synthetic or published values (CNS check digit,
             cadastral entropy, Fellegi–Sunter weights) and their README
```

## Reproduce

These run standalone from the repository root (tested with Python 3.12, pandas
2.2.3 and matplotlib 3.10.0; `validate-docbr` 2.0.1 for the CNS script):

```bash
pip install pandas matplotlib validate-docbr

python paper/gerar_tex_final.py                     # numbers/numbers.json -> paper/ml4h_final.tex
python analysis/ml4h_figuras_finais.py --sem-km     # outputs/*.csv -> paper/figures/ (all but figC)
python synthetic/validate_synthetic_cns.py && python synthetic/entropy_and_fs_example.py
```

The figure script redraws the nine CSV-driven figures byte-identical to those in
`paper/figures/`. The Kaplan–Meier figure (`figC_km_scenarios`) is drawn from the
patient-level cohort and is not redrawn here.

To compile the paper (TeX Live or Overleaf):

```bash
cd paper && pdflatex ml4h_final && bibtex ml4h_final && pdflatex ml4h_final && pdflatex ml4h_final
```

The numbers themselves can also be re-derived from the tables, which rewrites
`numbers/numbers.json` with identical content:

```bash
python analysis/coletar_numeros.py && python numbers/make_provenance.py
```

Compiling the PDF needs the ML4H 2026 LaTeX template (`jmlr.cls` and its style
files), which is not included.

### What does not run standalone

The 21 analysis scripts document exactly how each aggregate was computed from the
public DATASUS/INCA microdata. They run inside the authors' ETL and linkage
pipeline, which is not released, and are not standalone: they read that
pipeline's intermediate files (Parquet extracts of the raw files, DuckDB caches of
the outpatient base, the linked cohorts) and import its internal modules — `_raiz`
and `paths` (repository root and paths), `aux_cns` (CNS check digit as a DuckDB
macro), `aux_recuperacao` (Wilson interval), the `c50` package (`c50.config`,
`c50.analises`; breast-cancer cohort), `gerar_relatorio_wilms` (nephroblastoma
cohort), `validar_cns_bi`, `analisar_cns_faltante_bi`, `cascade_bi_estabilidade`,
`score_pares_candidatos_bimodal`, `monte_carlo_sensibilidade_linkage` and
`nefroblastoma_ipw_cenario_b`. Where a script imports another script that is
here (`cenarios_degradacao_linkage`, `propagacao_erro_linkage`,
`c50_calibracao_rotulos`), the dependency is included. Each script copies its
aggregate CSVs to `<pipeline root>/outputs` (the `AGREGADOS` variable), which is
`outputs/` in this repository. In these copies, that folder and the variable
were renamed from the authors' internal layout. Comments and docstrings are in
Portuguese.

## Paper section or figure → script → output

Tag prefixes refer to `numbers/provenance.md`. Scripts are in `analysis/`, tables
in `outputs/`.

| Paper | Tags | Script | Output file(s) |
|---|---|---|---|
| E1 identifier reliability; appendix National Audit | `REG-*` | `tabela1_cns_nacional.py` | `tabela1_nacional_ate2025.csv` |
| E1 / appendix, unreliable CNS by race, year-standardised | `E-YEAR-*` | `tabela1_cns_padronizada_ano.py` | `tabela1_raca_padronizada_ano_ate2025.csv` |
| E1 / appendix, cadastral entropy | `F-H-*`, `F-MUN`, `F-DOB`, `MULTI-*` | `entropia_miller_madow_decomposicao.py` | `entropia_miller_madow_ate2025.csv`, `decomposicao_identidades_ate2025.csv` |
| E1 / appendix, collisions against chance | `F-CHANCE*`, `F-TRINCAS-*` | `colisao_trinca_baseline_acaso.py` | `colisao_trinca_baseline_ate2025.csv` |
| Appendix, candidate-pair score | `F-ORDER`, `F-UMUN-UF` | `score_pares_rescore_intra_uf.py` | `distribuicao_score_intraUF_ate2025.csv`, `u_m_empirico_por_uf_ate2025.csv` |
| E2 cascade with the CNS hidden; appendix Cascade Test | `A-*` (except `A-R1/3/4`, `A-RHC`) | `cascata_registro_retido_uf.py` | `cascata_registro_retido_ate2025.csv` |
| Appendix, recall per record | `A-R1`, `A-R3`, `A-R4` | `cascade_bi_recall_interno.py` | `recall_chaves_ate2025.csv` |
| Appendix, blocking precision | `BLK-1IN` | `completude_fnr_parity.py` | `metricas_por_estrato_ate2025.csv` |
| E2 / appendix Fellegi–Sunter Parameters | `D-*` | `fs_diagnostico_u_m.py` | `fs_u_corrigido_coorte.csv`, `fs_u_corrigido_nacional.csv`, `fs_m_conjunto_nacional.csv`, `fs_duplicatas_resumo_coorte.csv`, `fs_u_sem_duplicatas_coorte.csv` |
| E3 recall and FNR by race, region, completeness | `REC-*` | `completude_fnr_parity.py` | `metricas_por_estrato_ate2025.csv` |
| E3 missing race, adjusted; cluster CIs | `E-RAW`, `E-ADJ`, `E-FAC*`, `E-BLQ`, `E-REC-*`, `F-CI*`, `F-WIDEN-*` | `completude_modelo_ajustado.py` | `efeito_raca_faltante_ate2025.csv`, `paridade_fnr_cluster_ate2025.csv` |
| E4 nephroblastoma, degradation scenarios | `S0-*`, `S1-*`, `S1B-*`, `S2-*` | `cenarios_degradacao_linkage.py` | `tabela_cenarios_degradacao.csv` |
| E4 propagation of measured link loss | `B-*` | `propagacao_erro_linkage.py` | `propagacao_conjuntos.csv`, `propagacao_taxas_medidas.csv`, `propagacao_gap_racial.csv`, `propagacao_limiares.csv` |
| E4 death sources against the registry | `OB-*`, `RX-*`, `SIH-*`, `A-RHC` | `concordancia_rhc_sus.py` | `tabela_rhc_x_sus.csv`, `tabela_obitos_c1_por_fonte.csv`, `tabela_sensibilidade_sih.csv` |
| E4 / appendix, capture–recapture and registry-only deaths | `CR-*`, `SORHC-*`, `SIMLINK-*`, `MUNROUTE-SORHC`, `BOTH-INSIM-PCT` | `captura_recaptura_obitos.py` | `captura_recaptura_obitos.csv`, `captura_recaptura_cenarios.csv`, `decomposicao_so_rhc.csv` |
| E4 / appendix, the registry link | `RHC-*` | `validacao_rhc_surrogate.py`, `sensibilidade_match_rhc.py` | `validacao_rhc_resumo.csv`, `tabela_sensibilidade_match_rhc.csv` |
| E4 / appendix, race as the missing variable | `F-CC*`, `F-IPW*`, `F-MI*`, `F-IMP`, `F-W-*`, `GAP-CI` | `tabela2_raca_faltante.py` | `tabela2_revisada.csv`, `tabela2_pesos_ipw.csv` |
| E5 breast-cancer Cox model under three death labels | `C-*` (except `C-INJ*`) | `c50_calibracao_rotulos.py` | `calibracao_rotulos_hr.csv`, `calibracao_rotulos_global.csv`, `calibracao_rotulos_subgrupos.csv`, `calibracao_rotulos_populacao.csv` |
| E5 measured link loss injected into the model | `C-INJ*` | `c50_injecao_fnr.py` | `injecao_fnr_c50.csv` |
| E5 / appendix, deaths by source and stage | `U-*`, `C50-C1*`, `C50-C2*`, `C50-RATIO-*` | `c50_obitos_por_fonte_estadio.py` | `obitos_por_fonte_estadio.csv`, `obitos_pipeline_por_camada.csv` |
| E5 / appendix, registry-only deaths searched in SIM | `C50-BOTH-*`, `C50-SORHC-*`, `C50-FALSOS*`, `C50-CR-MORT` | `c50_decomposicao_so_rhc.py` | `decomposicao_so_rhc_c50.csv` |
| Fig. 1 (`fig1_overview`), Fig. 2 (`fig2_sources_links`) | — | `ml4h_figuras_finais.py` (schematics) | — |
| Fig. 3 (`fig3_cascade_pools`) | | `ml4h_figuras_finais.py` ← `cascata_registro_retido_uf.py` | `cascata_registro_retido_ate2025.csv` |
| Fig. 4 (`fig4_race_recall`) | | ← `completude_fnr_parity.py`, `completude_modelo_ajustado.py` | `metricas_por_estrato_ate2025.csv`, `efeito_raca_faltante_ate2025.csv` |
| Fig. 5 (`fig5_death_budget`) | | ← `concordancia_rhc_sus.py`, `propagacao_erro_linkage.py` | `tabela_obitos_c1_por_fonte.csv`, `tabela_rhc_x_sus.csv`, `propagacao_conjuntos.csv` |
| Fig. 6 (`fig6_scenarios`) | | ← `cenarios_degradacao_linkage.py`, `propagacao_erro_linkage.py`, `captura_recaptura_obitos.py` | `tabela_cenarios_degradacao.csv`, `propagacao_taxas_medidas.csv`, `captura_recaptura_cenarios.csv` |
| Fig. 7 (`fig7_model_labels`) | | ← `c50_calibracao_rotulos.py` | `calibracao_rotulos_hr.csv`, `calibracao_rotulos_subgrupos.csv` |
| Appendix figure `figB_propagation` | | ← `propagacao_erro_linkage.py`, `captura_recaptura_obitos.py`, `cenarios_degradacao_linkage.py`, `completude_fnr_parity.py` | `propagacao_varredura.csv`, `propagacao_gap_racial.csv`, `propagacao_taxas_medidas.csv`, `propagacao_conjuntos.csv`, `captura_recaptura_cenarios.csv`, `tabela_cenarios_degradacao.csv`, `metricas_por_estrato_ate2025.csv` |
| Appendix figure `figC_km_scenarios` | | ← `cenarios_degradacao_linkage.py` (patient-level cohort) | not released |
| Appendix figure `figD_calibration` | | ← `c50_calibracao_rotulos.py` | `calibracao_rotulos_global.csv` |

## Data and ethics

The source files are public-access open data. They contain record-level date of
birth, sex and municipality, and the CNS is released encrypted, so the data are
pseudonymised, not anonymous. Brazilian research-ethics norms exempt research that
uses only public-access information from ethics-committee registration
(Resolução CNS 510/2016, Art. 1, sole paragraph, items II and III). We attempted
no re-identification and linked no source outside these files. This repository
contains aggregates only: every table in `outputs/` is a count, rate, estimate or
interval by stratum or scenario. A pair-level table produced during the analysis
(pairs of distinct encrypted CNS sharing birth date, sex and municipality) is not
included; the paper cites only its aggregate summary
(`outputs/fs_duplicatas_resumo_coorte.csv`).

## License and citation

MIT License (see `LICENSE`).

```bibtex
@inproceedings{freitas2026auditing,
  title     = {Auditing Identity Noise and Linkage Degradation at National Scale:
               From Bias Measurement to Downstream Clinical Distortion},
  author    = {Freitas, Jo{\~a}o Luis Reis and {da Silva}, Leonardo Dexheimer and
               Berg, Felipe Matias and Wolosker, Nelson},
  booktitle = {Machine Learning for Health (ML4H) 2026, Findings Track},
  year      = {2026}
}
```
