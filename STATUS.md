# Project status

Last updated: 2026-10-21. This page gives the approved seminar topic, the decisions taken, the team roles, the work plan with dates, the supplementary historical analysis and the open items. Figures marked preliminary are not yet reviewed.

## 1. Seminar topic (revised 21 October 2026: financial appraisal of the park-only operator ledger)

**Title.** Scenario-Based Financial Feasibility and Break-Even Analysis of a Proposed Disney-Type Theme Park in Thailand.

**Scope.** Financial appraisal of the park-only operator ledger. The capital boundary is the park-only proposal of about THB 200 billion (not verified), matched to the theme-park revenue. The THB 300 billion sports and entertainment complex is kept only as a reconciliation to the earlier model.

**Out of scope.** The national economic appraisal, the fiscal flows of public participation, and tourism demand and receipts as research questions. Visitor numbers, spending and displacement are inputs to the operator ledger, not hypotheses. The historical evidence on openings elsewhere is out of scope (section 5).

**Research question.** Is a proposed Disney-type theme park in Thailand financially viable for an operator at the documented cost of capital, what break-even attendance and capital cost does it need, and how sensitive is that to costs, revenue and opening delay?

**Objectives.**
1. Estimate the operator's financial net present value, internal rate of return and break-even attendance (H1).
2. Estimate the break-even capital cost of the park-only ledger (H1).
3. Report the reconciliation to the earlier 300 bn model, and the tax and replacement-capex sensitivities.
4. Test the results under the sensitivity and switching cases (section 7.4 of the Phase 1 document).

**Hypotheses.**
- **H1.** The theme park has a positive financial net present value after tax under the documented base-case assumptions.
- **H2 and H3 retired (20 and 21 October 2026).** The economic and fiscal hypotheses are not tested in this frame.

**Method.** A bottom-up appraisal that follows the ADB (2017) Guidelines for the Economic Analysis of Projects and the NESDC (2024) Guide to Public Investment Project Appraisal.
- **Operator ledger:** revenue less cash operating costs, capital, replacement capex and corporate income tax; financial NPV, IRR, break-even capital cost and break-even attendance. The earlier national and fiscal ledgers are not used.
- **Inputs:** demand (visitors, spending, capacity), costs, financing terms, tax rates and discount rates, each labelled by source status.
- **Scenarios:** low, base and high cases, break-even attendance, and stress tests on cost, revenue and opening delay.

## 2. Decisions taken (14 October 2026)
- The title, the objectives and hypotheses were approved on 14 October 2026. On 19 October 2026 the scope was narrowed to finance economics, and the hypotheses are restated as H1 to H3 (section 1).
- The team roles in section 3 are confirmed.
- The primary scope was the THB 300 billion complex until 21 October 2026. From 21 October 2026 the primary boundary is the park-only operator ledger (THB 200 billion proposal, matched to the theme-park revenue). The complex is kept as a reconciliation in `seminar/workbook/Phase1_financial_appraisal_v2.xlsx`.
- 20 and 21 October 2026: Financial Appraisal only (decision `decision-financial-only-20261020`); cash operating costs as the primary basis; corporate income tax at 20 percent; replacement capex zero in the base case, because no source exists.
- 21 October 2026, decided by kamolpeen with the same weight as team approval: on-site income at the sourced 1,862 THB per operator visit is part of the primary revenue (primary NPV -144.05 bn before tax; the ticket-only result, -163.12 bn, is a memo); the HK depreciation proxy for replacement capex is not appropriate as a base and is kept only as a labelled stress; the attendance benchmark table (Magic Kingdom 17.8 million in 2024) is added to the Phase 1 document; a dated note is added to the 19 October final-results record.
- The 13F investor-horizon project is removed from the repository and from the local working folder. Its history stays in git.
- Tourism demand and receipts, and the historical analysis of Hong Kong and Shanghai, are out of scope from 19 October 2026. They are kept in the repository as a record.

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
| 19 Oct | Revise the Phase 1 document to the finance economics scope and re-export it | Peem, chow | Revised Phase 1 document |
| 19 Oct | Confirm all assumptions in the assumptions table | Team | All statuses record the confirmation |
| From 20 Oct (estimate) | Source the remaining pending inputs that the ledgers use (8 rows, each with a stated reason) | Team | Input table |
| From 20 Oct (estimate) | Complete the three ledgers with break-even, IRR and payback outputs, with tests | Peem, with an agent | Model and tests |
| From 27 Oct (estimate) | Stress tests following ADB (2017): cost, revenue and delay; switching values | Team | Results |
| November (estimate) | Final report and presentation | Team | Final report |

The dates after 16 October are estimates. They will be confirmed after the Phase 1 review.

## 5. Out of scope since 19 October 2026: historical analysis (kept as a record)

Folder: `disney-tourism-thailand`. Question: did the Hong Kong and Shanghai openings change international tourism receipts relative to GDP, and what do the openings point to for a complex of the proposed size? The full closure record is `seminar/SUPPLEMENT_CLOSURE_RECORD.md`. Appendix A of the Phase 1 document contains the same text.

- **Sample:** 44 openings (episodes), 27 feasible, 19 in the primary sample, in 19 economies. The importance route needs at least 20 episodes, so the pre-registered rule chose the ambient route.
- **Estimator:** the scaling law in cost share (A2) is selected by the leave-one-economy-out rule. For the THB 300 billion complex the median effect is +0.40 percentage points of GDP (5 to 95 percent: -1.14 to +1.94). At the 2025 baseline that is +US$2.33 billion a year (-6.55 to +11.17); at the 2019 baseline it is +US$2.19 billion (-6.18 to +10.53). The 90 percent interval for a new study, built on the errors of A2, runs from -1.52 to +2.33 percentage points and includes zero. The weighted-transport interval, a diagnostic, runs from -2.73 to +1.27.
- **Transport diagnostic:** the importance-weighted optimal transport gives -0.73 points, with a bootstrap band of -1.28 to -0.19. Its validation fails the 0.95 limit against the uniform-weight transport (ratio 1.009), and its importance model is not usable. It is a diagnostic, not an effect.
- **Reading:** the data cannot separate the effect from zero. The historical results are not the Thai net benefit.
- **Open items for this track:** section 6 of the closure record.

## 6. Repository and reproducibility

- All six notebooks run on the dataset in the repository. The first run of 16 October 2026 took 78 minutes for notebooks 00 to 05, of which notebook 03 took 75 minutes. The re-run of the same day took 3997 seconds, of which notebook 03 took 3778 seconds.
- The tests run with `python -m pytest -q` from `disney-tourism-thailand` and take about 7 minutes.
- Generated files in `results/` and `figures/` change only through a re-run pull request, as described in CONTRIBUTING.md.

## 7. Open items

- Thai inputs (inputs.csv, 131 rows): 87 sourced. Of these, 44 come from Thai data sources, and the rest are guidance values, illustrative examples and comparators outside Thailand. 9 are unconfirmed (candidate or secondary values, flagged and not used by the model). 8 are pending or blank, each with a stated reason. 1 is dropped from use (the Bank of Thailand receipts row). 9 are team decisions, 7 are claims or proposal figures, 4 are derived, and 6 are proxies or generalized values. The model reads the sourced names, the proposal capital cost and the labelled assumptions table, and no other input.
- All assumptions in the assumptions table were confirmed by the team on 19 October 2026.
- Team decisions confirmed by the leader on 14 October 2026, recorded as labelled assumptions: WACC 8 percent base (10 and 7 percent in the low and high cases); project life 20 years base (20 and 25 years in the low and high cases); attendance 7.5 million; incremental share 30 percent; displacement 30 percent; domestic retention 50 percent; spending-per-visit base 10,000 THB and high case 20,000 THB (the low case is the sourced on-site benchmark, below); occupancy ceiling 85 percent base. None is a sourced value. The observed 80.42 percent would remove all base headroom and is reported as the alternative.
- Spending per visit, decided 16 October 2026: the low case is the sourced on-site benchmark of 1,862 THB (Oriental Land's Tokyo Disney Resort, FY3/2026, 8,796 yen per guest, converted at the Bank of Thailand JPY mid rate of 12 October 2026, 21.1734 THB per 100 yen). The base of 10,000 THB and the high case of 20,000 THB are kept. The per-trip figures (TAT 50,900 THB; MoTS 48,580 THB) are not per park visit and are not used.
- Operating costs include depreciation, by the leader's decision, in the earlier confirmed basis. From 21 October 2026 the primary operator ledger uses cash operating costs with depreciation excluded, decided by kamolpeen on 21 October 2026, with the same weight as team approval. The confirmed basis is kept as a reconciliation.
- Sources conflict on the high-speed rail opening year (2031 from the EEC Office; 2029 in the Wikipedia infobox and 2030 in its later status entry; not used). Chon Buri foreign visitors in 2024 are 11,689,827 (TAT workbook); the figure of 28,156,302 is a Thai plus foreign total and is not used.
- U-Tapao has no sourced calendar 2019 figure, because AOT does not operate it. The calendar 2024 figure is the derived 384,881 passengers from the Ministry open-data file, and the draft uses it. The operator's budget-year figures are labelled comparators: 370,210 for October 2023 to September 2024 (sourced), and 1,665,004 for 2019 (proxy).
- References: Chan (2004) is verified. Plaza (2006) is read in full from the attached copy and is cited without a link. Tan, How and Koh is not verified and is not cited. Milman (2010) is removed: its full text is paywalled, and its author list conflicts (one author in the publisher's author section; three in the citation line and the UCF STARS record). Abadie (2021), Burgan and Mules (2001), Cameron and Miller (2015) and Dwyer et al. (2000) are removed as paywalled and unread.
- The changes of 16 October 2026 are committed in four commits: code `0197a1f`, inputs `15a5ff1`, the re-run results `a811b0b`, and the record and documents that follow them. The decisions are listed in section 6 of `seminar/SOURCING_LOG_2026-10-16.md`.
- The origin of `data/raw/disney_did_panel.csv` is unknown. `DATA_NOTICE.md` gives the World Bank attribution and asks the project owner to state the origin of the panel file.

## 8. Working documents

- `seminar/PHASE1_DRAFT.md`: the Phase 1 document in working form, with PENDING markers for the team.
- `seminar/TEAM_BRIEFS.md`: the aligned brief for each role, with the corrections and the checks.

## 9. Re-run of 16 October 2026

- The notebooks were re-run on the dataset with the code in the pull request "Remove the simulated world from the pipeline". The pipeline has no simulated mode. The simulation generators and the tests that used them are removed. The tests that needed a generated world now use hand-built fixtures in `tests/handmade_panels.py` and `tests/handmade_world.py`.
- The re-run of 16 October 2026 (the second re-run) changed the notebook 05 outputs: the interval for a new study is built on the A2 errors, and the placebo nulls are used without rescaling. Its results are in `results/`, its record is in `seminar/provenance/rerun_2026-10-16/`, and the commit message names the code commit and the run time.
- The full test suite passes on the code of the re-run: 1879 passed and 1 skipped. The dataset check (`DTT_RUN_REAL=1`) passed in the first run of 16 October 2026.
- The route rule selects the ambient route. The importance model is not usable, the weighted transport does not beat the uniform transport by the 0.95 margin, the target is supported, and A2 is selected. Section 5 gives the effect.
- The Haswell and Zen kernels pass. The SkylakeX kernel cannot run on the development computer.

## How you can help

- Read the topic and hypotheses above, then comment in the group.
- Claim a task in a GitHub Issue (template "Task") so that two people do not work on the same thing.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md): work on a branch, open a pull request, and do not hand-edit generated results.


