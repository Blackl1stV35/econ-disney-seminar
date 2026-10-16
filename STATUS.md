# Project status

Last updated: 2026-10-16. This page gives the approved seminar topic, the decisions taken, the team roles, the work plan with dates, the supplementary historical analysis and the open items. Figures marked preliminary are not yet reviewed.

## 1. Seminar topic (approved 14 October 2026)

**Title.** Appraising a Proposed Disney-type Theme-Park Complex in Thailand: Bottom-up Tourism, Financial, Fiscal and Economic Returns under Scenario Analysis

**Scope.** The THB 300 billion sports and entertainment complex, which is the high-expectation scope selected for the appraisal. The park-only concepts of THB 100 to 200 billion are different projects with different demand, revenue and cost boundaries. They are not mixed with the complex and are not modelled in the primary appraisal.

**Research question (draft).** What are the potential tourism and economic impacts of a Disney-type complex in Thailand, and does the project create net value for Thailand once displacement, leakage and public costs are counted?

**Objectives.**
1. Estimate the incremental international visitor expenditure that the complex generates in Thailand, relative to a no-project baseline (H1).
2. Estimate the operator's financial net present value, internal rate of return and break-even attendance (H2).
3. Estimate the national economic net present value after displacement, leakage and resource costs (H3).
4. Estimate the fiscal net flows of public participation, as a supplementary check (H4).

**Hypotheses.**
- **H1.** The complex generates positive incremental international visitor expenditure in Thailand relative to the no-project scenario, after spending displaced from other Thai destinations is subtracted.
- **H2.** The complex has a positive financial net present value under the documented Thai base-case demand and cost assumptions.
- **H3.** The complex produces a positive economic net present value for Thailand after incremental resource costs, displaced activity and material social costs are accounted for.
- **H4 (supplementary).** The discounted incremental tax and fee revenue attributable to the project exceeds its discounted project-specific public expenditure and fiscal costs under the stated policy scenario.

**Method.** A bottom-up appraisal that follows the ADB (2017) Guidelines for the Economic Analysis of Projects and the NESDC (2024) Guide to Public Investment Project Appraisal.
- **Demand:** visitor segments (Thai residents, international short-haul, international long-haul, repeat visitors), incremental attendance, length of stay, per-visitor spending, opening ramp-up and capacity.
- **Netting:** spending displaced from other Thai destinations, imported inputs, profit repatriation, opportunity cost of land, labour and capital, congestion and environmental costs.
- **Three ledgers:** operator (financial NPV, IRR, payback), fiscal (net flows) and national economic (economic NPV, benefit-cost ratio).
- **Scenarios:** low, base and high cases, break-even attendance, incremental share and capex thresholds, and stress tests on opening delay and displacement.
- **Historical evidence** (Hong Kong, Shanghai, Singapore and the international literature) provides calibration ranges and qualitative benchmarks. It does not determine the Thai net benefit directly.
- **Cross-check:** a reference-class predictor from the replacement recommendation is used only as a benchmark against the historical cases, never as the net-benefit estimate.

## 2. Decisions taken (14 October 2026)
- The title, the objectives and hypotheses H1 to H3 are approved. H4 is supplementary.
- The team roles in section 3 are confirmed.
- The primary scope is the THB 300 billion complex.
- The 13F investor-horizon project is removed from the repository and from the local working folder. Its history stays in git.
- The historical econometric analysis of Hong Kong and Shanghai is supplementary.

## 3. Team and roles

| Member | Role |
| --- | --- |
| Peem | Leader; final method; the supplementary historical analysis |
| chow | Hypotheses; integration of the sections; contact with the professor |
| tap | Research questions |
| Tai | Literature; check of the Singapore literature review |
| LINK | Research methods |

Work is coordinated through GitHub Issues, using the template "Task", as described in CONTRIBUTING.md.

## 4. Work plan

| Date | Task | Owner | Output |
| --- | --- | --- | --- |
| 14 Oct | Send the reframed topic with objectives to the professor | Peem, chow | Topic submitted |
| 14 to 15 Oct | Literature: ADB and NESDC method sections; Hong Kong, Shanghai and Singapore as context | Tai, chow | Literature section |
| 15 Oct | Research questions and objectives | tap | Research question section |
| 15 Oct | Hypotheses H1 to H3 (H4 supplementary) | chow | Hypothesis section |
| 15 Oct | Method outline, scenario grid and break-even definition | LINK, with Peem | Method section |
| 15 Oct | Integrate the sections and check consistency | chow, Peem | Draft |
| 16 Oct | Submit the Phase 1 seminar document | Peem | Submitted |
| From 19 Oct (estimate) | Source the Thai inputs: segment mix, attendance ramp, spending, capex, operating costs, tax rates, discount rate and shadow parameters from NESDC (2024) | Team | Input table: 58 sourced values, 29 unconfirmed, 13 blank (see section 7) |
| From 19 Oct (estimate) | Build the appraisal model in `disney-tourism-thailand`, with tests for the three ledgers, the scenario switch and break-even | Peem, with an agent | Model with labelled cases, the NESDC sensitivity table and tests; the appraisal and notebook tests pass. The NESDC values are sourced; WACC and project life are team recommendations, and the other team decisions are labelled |
| From 26 Oct (estimate) | Stress tests following ADB (2017), sensitivity on displacement and incremental share, limitations and decision gates | Team | Results |
| November (estimate) | Final report and presentation | Team | Final report |

The dates after 16 October are estimates. They will be confirmed after the Phase 1 review.

## 5. Supplementary: historical analysis (preliminary)

Folder: `disney-tourism-thailand`. Question: did the Hong Kong and Shanghai openings change international tourism receipts relative to GDP, and what do the openings point to for a complex of the proposed size? The full closure record is `seminar/SUPPLEMENT_CLOSURE_RECORD.md`. Appendix A of the Phase 1 document contains the same text.

- **Sample:** 44 openings (episodes), 27 feasible, 19 in the primary sample, in 19 economies. The importance route needs at least 20 episodes, so the pre-registered rule chose the ambient route.
- **Estimator:** the scaling law in cost share (A2) is selected by the leave-one-economy-out rule. For the THB 300 billion complex the median effect is +0.40 percentage points of GDP (5 to 95 percent: -1.14 to +1.94). At the 2025 baseline that is +US$2.33 billion a year (-6.55 to +11.17); at the 2019 baseline it is +US$2.19 billion (-6.18 to +10.53). The 90 percent interval for a new study runs from -2.73 to +1.27 percentage points and includes zero.
- **Transport diagnostic:** the importance-weighted optimal transport gives -0.73 points, with a bootstrap band of -1.28 to -0.19. Its validation fails the 0.95 limit against the uniform-weight transport (ratio 1.009), and its importance model is not usable. It is a diagnostic, not an effect.
- **Reading:** the data cannot separate the effect from zero. The historical results are not the Thai net benefit.
- **Open items for this track:** section 6 of the closure record, and the later phases listed there.

## 6. Repository and reproducibility

- All six notebooks run on the dataset in the repository. The run of 16 October 2026 took 78 minutes for notebooks 00 to 05, of which notebook 03 took 75 minutes.
- The tests run with `python -m pytest -q` from `disney-tourism-thailand` and take about 7 minutes.
- Generated files in `results/` and `figures/` change only through a re-run pull request, as described in CONTRIBUTING.md.

## 7. Open items

- Thai inputs (inputs.csv, 127 rows): 58 sourced; 29 unconfirmed (candidate or secondary values, flagged and not used by the model); 13 blank, each with a stated reason (not found, withdrawn, conflicting, a text row, or multi-value). Nine team decisions, five labelled proxies and three derived values fill other rows, so the model reads them only through the labelled assumptions table.
- Team decisions confirmed by the leader on 14 October 2026, recorded as labelled assumptions: WACC 8 percent base (10 and 7 percent in the low and high cases); project life 20 years base (20 and 25 years in the low and high cases); attendance 7.5 million; incremental share 30 percent; displacement 30 percent; domestic retention 50 percent; spending-per-visit placeholders of 5,000, 10,000 and 20,000 THB; occupancy ceiling 85 percent base. None is a sourced value. The observed 80.42 percent would remove all base headroom and is reported as the alternative.
- Still open within the confirmed decisions: the basis for spending per visit. The per-trip figures (TAT 50,900 THB; MoTS 48,580 THB) are not per park visit, so the placeholders are not yet sourced.
- Operating costs include depreciation, by the leader's decision. The overlap with capital cost is kept; the anchors have not been restated.
- Sources conflict on the high-speed rail opening year (2031 from the EEC Office; 2029 in the Wikipedia infobox and 2030 in its later status entry; not used). Chon Buri foreign visitors in 2024 are 11,689,827 (TAT workbook); the figure of 28,156,302 is a Thai plus foreign total and is not used.
- U-Tapao has no sourced calendar 2024 or calendar 2019 figure, because AOT does not operate it. The table uses the operator's budget-year figures (370,210 for 2024; 1,665,004 for 2019) as labelled proxies.
- References: Chan (2004) is verified. Plaza (2006) is read in full from the attached copy and is cited without a link. Tan, How and Koh is not verified and is not cited. Milman (2010) is removed: its full text is paywalled, and its author list conflicts (one author in the publisher's author section; three in the citation line and the UCF STARS record). Abadie (2021), Burgan and Mules (2001), Cameron and Miller (2015) and Dwyer et al. (2000) are removed as paywalled and unread.
- Changes not yet committed: the model, input and assumption tables, tests, the benchmark note, the Phase 1 draft and this status page. Commit after the leader's go-ahead.
- The origin of `data/raw/disney_did_panel.csv` is unknown. `DATA_NOTICE.md` gives the World Bank attribution and asks the project owner to state the origin of the panel file.

## 8. Working documents

- `seminar/PHASE1_DRAFT.md`: the Phase 1 document in working form, with PENDING markers for the team.
- `seminar/TEAM_BRIEFS.md`: the aligned brief for each role, with the corrections and the checks.

## 9. Re-run of 16 October 2026

- The notebooks were re-run on the dataset with the code in the pull request "Remove the simulated world from the pipeline". The pipeline has no simulated mode. The simulation generators and the tests that used them are removed. The tests that needed a generated world now use hand-built fixtures in `tests/handmade_panels.py` and `tests/handmade_world.py`.
- The new results are in `results/` and `figures/`, in the second pull request, which exists only for the re-run and names the commit and the run time, as CONTRIBUTING.md asks.
- The full test suite passes on the code branch: 1877 passed and 1 skipped. The dataset check (`DTT_RUN_REAL=1`) passed in the run of 16 October 2026.
- The route rule selects the ambient route. The importance model is not usable, the weighted transport does not beat the uniform transport by the 0.95 margin, the target is supported, and A2 is selected. Section 5 gives the effect.
- The Haswell and Zen kernels pass. The SkylakeX kernel cannot run on the development computer.

## How you can help

- Read the topic and hypotheses above, then comment in the group.
- Claim a task in a GitHub Issue (template "Task") so that two people do not work on the same thing.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md): work on a branch, open a pull request, and do not hand-edit generated results.


