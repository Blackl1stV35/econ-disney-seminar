# Project status

Last updated: 2026-10-14. This page gives the approved seminar topic, the decisions taken, the team roles, the work plan with dates, the supplementary historical analysis and the open items. Figures marked preliminary are not yet reviewed.

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
| From 19 Oct (estimate) | Source the Thai inputs: segment mix, attendance ramp, spending, capex, operating costs, tax rates, discount rate and shadow parameters from NESDC (2024) | Team | Input table with a source for every value |
| From 19 Oct (estimate) | Build the appraisal model in `disney-tourism-thailand`, with tests for the three ledgers, the scenario switch and break-even | Peem, with an agent | Model, notebook and tests |
| From 26 Oct (estimate) | Stress tests following ADB (2017), sensitivity on displacement and incremental share, limitations and decision gates | Team | Results |
| November (estimate) | Final report and presentation | Team | Final report |

The dates after 16 October are estimates. They will be confirmed after the Phase 1 review.

## 5. Supplementary: historical analysis (preliminary)

Folder: `disney-tourism-thailand`. Question: did the Hong Kong and Shanghai openings change international tourism receipts relative to GDP?

- **Sample:** 44 openings (episodes), 27 feasible, 19 in the primary sample, in 19 economies. The importance route needs at least 20 episodes, so the pre-registered rule chose the ambient route.
- **Estimator:** the scaling law in cost share (A2) was selected. For the THB 300 billion complex it gives a median of about +US$2.2 billion a year (+0.4 percent of GDP). The 90 percent interval runs from about -US$6.2 to +US$10.5 billion and includes zero.
- **Reading:** the data cannot separate the effect from zero. The historical results are not the Thai net benefit.
- **Open items for this track:** 37 reference values of the original Stata analysis that notebook 01 does not reproduce (1179 of 1216 match); the data notice; the check of the 20 World Bank files against the official API; the final verification and an independent review.

## 6. Repository and reproducibility

- All six notebooks run on the dataset in the repository. A full run takes one to two hours on a laptop, almost all of it in notebook 03.
- The tests run with `python -m pytest -q` from `disney-tourism-thailand` and take about 13 minutes.
- Generated files in `results/` and `figures/` change only through a re-run pull request, as described in CONTRIBUTING.md.

## 7. Open items

- Verify the references in the replacement recommendation before they appear in the Phase 1 document: Milman, Okumus and Dickson (2010), Plaza (2006), Burgan and Mules (2001), Dwyer et al. (2000), Abadie (2021) and Cameron and Miller (2015). They were not verified in the session that produced the recommendation.
- Read NESDC (2024) and ADB (2017) for the discount rate, the shadow parameters and the appraisal criteria.
- Source the Thai inputs. No government document was found that states the assumptions behind the projection of Thai GDP.
- Confirm the proposal parameters (capex, site, operator and timing) with the professor.
- Data notice: the origin of `data/raw/disney_did_panel.csv` and the World Bank attribution (CC BY 4.0).
- Check of the 20 World Bank files against the official API (a script is still to be written).

## How you can help

- Read the topic and hypotheses above, then comment in the group.
- Claim a task in a GitHub Issue (template "Task") so that two people do not work on the same thing.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md): work on a branch, open a pull request, and do not hand-edit generated results.
