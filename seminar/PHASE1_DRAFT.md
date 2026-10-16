# Phase 1 seminar document (working draft, 14 October 2026)

Status: working draft for the team. Items marked PENDING need input from the team before submission. Submission deadline: 16 October 2026.

## 1. Title

Appraising a Proposed Disney-type Theme-Park Complex in Thailand: Bottom-up Tourism, Financial, Fiscal and Economic Returns under Scenario Analysis

## 2. Research question and sub-questions

**Primary question.** What are the potential tourism and economic impacts of a Disney-type complex in Thailand, and does the project create net value for Thailand once displacement, leakage and public costs are counted?

1. How much incremental international visitor spending would the complex generate, relative to a no-project baseline?
2. Is the complex financially viable for an operator at the documented cost of capital, and what break-even attendance does it need?
3. Does the complex create positive economic net value for Thailand at the national social discount rate?
4. What are the fiscal flows of public participation, and how sensitive are the results to costs, revenue and opening delay?

## 3. Objectives

1. Estimate incremental international visitor expenditure in Thailand against a no-project baseline, net of displaced spending (H1).
2. Estimate the operator's financial net present value, internal rate of return and break-even attendance (H2).
3. Estimate the national economic net present value after displacement, leakage and resource costs (H3).
4. Estimate the fiscal net flows of public participation as a supplementary check (H4).

## 4. Hypotheses

- **H1.** The complex generates positive incremental international visitor expenditure in Thailand relative to the no-project scenario, after spending displaced from other Thai destinations is subtracted.
- **H2.** The complex has a positive financial net present value under the documented Thai base-case demand and cost assumptions.
- **H3.** The complex produces a positive economic net present value for Thailand after incremental resource costs, displaced activity and material social costs are accounted for.
- **H4 (supplementary).** The discounted incremental tax and fee revenue attributable to the project exceeds its discounted project-specific public expenditure and fiscal costs under the stated policy scenario.

**Decision criteria.** H1 holds if the base-case net incremental spending is positive after displacement. H2 holds if the financial net present value is positive at the financial cost of capital (WACC of 8 percent in the base case, recommended; see section 7.3). H3 holds if the economic net present value is positive at the social discount rate of 7 percent (NESDC 2024). H4 holds if the fiscal net present value is positive under the stated policy scenario.

## 5. Scope and scenarios

**Scope.** The primary appraisal covers the THB 300 billion sports and entertainment complex, the high-expectation scope of the Thai proposals. The park-only concepts of THB 100 to 200 billion are different projects with their own demand, revenue and cost boundaries. They are reported as separate cases only if time allows.

**Scenarios.** A base case, low and high demand cases, and the sensitivity cases of section 7.5. The sensitivity cases follow the NESDC (2024) guide: construction cost, operating cost and revenue each varied by 5, 10 and 20 percent, a one-year opening delay, and two-factor cases of cost and revenue together.

## 6. Literature

**(a) Appraisal standards.** The ADB (2017) guidelines set a minimum required economic internal rate of return of 9 percent for investment projects and 6 percent for social sector and environmental projects (paragraph 194), and require sensitivity analysis and switching values (Section VII). The NESDC (2024) guide recommends a social discount rate of 7 percent for appraisals begun in 2025 or later, within a range of 6.55 to 7.59 percent; the rate of 12 percent applies to appraisals begun before 2025. It sets a standard conversion factor of 0.7 for converting financial costs to economic costs where shadow prices cannot be computed.

**(b) Theme-park and single-attraction evidence.** Plaza (2006) estimates the effect of the Guggenheim Museum Bilbao on overnight stays, jobs and the return on investment. The published article reports 61,742 additional overnight stays a month from a single-series time-series model with no control group, so it supports context rather than a causal transfer. Published estimates for other attractions are reviewed in the causal benchmark note and are not transferred to Thailand (section 7.6).

**(c) Historical cases.** Hong Kong Disneyland and Shanghai Disney Resort are the closest cases. Their evidence is used for calibration ranges and as context, not as a transfer coefficient (section 7.6). Chan (2004) studies residents' perceptions of Hong Kong Disneyland and its impacts. The thesis reports two telephone surveys of 1,060 respondents and is verified against the PolyU repository record. A second case study, of Universal Studios Singapore by Tan, How and Koh, could not be verified and is not cited in this draft.

Methods references for the supplementary historical track are to be added once freely available copies are confirmed.

**Gap.** The literature provides single-attraction, macroeconomic and appraisal-standard evidence, but no bottom-up model for a Thai complex that follows the national appraisal guidance. This study fills that gap.

## 7. Method

**7.1 Framework.** A bottom-up appraisal that compares the with-project and without-project situations, following the ADB (2017) guidelines and the NESDC (2024) guide.

**7.2 Demand, spending and infrastructure.** The working hypothesis is that public infrastructure variation is the main driver of the tourism effect. The model therefore caps incremental visits by the infrastructure that can serve them, namely airport headroom and accommodation supply. Annual passenger movements convert to visits with `visits_from_passengers`, using two movements per round-trip visit, which is an assumption to be confirmed. Incremental visits are capped by headroom in `capped_net_spend`, and net spending follows from the capped visits after displacement.

Sourced values in the input table include U-Tapao terminal capacity of 3.7 million passengers a year (EEC Office, page status October 2025); the AOT six-airport total of 124,033,786 passengers in calendar 2024 and 100,058,695 in fiscal 2023; Suvarnabhumi and Don Mueang traffic for 2019 and 2024 (AOT); Chon Buri occupancy of 80.42 percent in 2024 (National Statistical Office, Table 17.1); Chon Buri foreign visitors of 11,689,827 in 2024 (TAT Intelligence Center); the Bank of Thailand exchange rate; and the Revenue Department tax rates. Hotel and guesthouse rooms of 956,372 refer to calendar 2021 (NSO Statistical Yearbook Thailand 2024, Table 17.4). Pattaya hotel keys of 31,581 in H2 2024 and 34,743 in H2 2025 are CBRE broker data, not an official source; the 2024 figure is used as a proxy for the room stock of the complex.

Team decisions, confirmed by the leader on 14 October 2026 and recorded as labelled assumptions in the assumptions table: WACC of 8 percent in the base case (10 and 7 percent in the low and high cases), and a project life of 20 years in the base case (20 and 25 years in the low and high cases). Attendance, incremental and displacement shares, domestic retention, the spending-per-visit placeholders and the 85 percent occupancy ceiling in the base case keep their working values. None is a sourced value. Inputs that could not be sourced are blank in the input table, and figures that were not confirmed are flagged unconfirmed, never sourced. The base case stops only when a sourced or proposal input is pending; team decisions do not stop it.
**7.3 Three ledgers.**
- **Operator ledger:** revenue minus operating costs and capital expenditure, giving the financial net present value, internal rate of return and break-even attendance at the financial cost of capital (WACC of 8 percent and a project life of 20 years in the base case, as recommended; the NESDC guide requires both to be stated and justified). Operating costs include depreciation, by the leader's decision.
- **Fiscal ledger:** tax and fee revenue from the net spending minus public capital and operating costs.
- **National economic ledger:** economic benefits minus economic costs, discounted at 7 percent. Economic costs use the conversion factor of 0.7 unless shadow prices are computed. The economic benefit is the net spending of the incremental visits, following ADB (2017) paragraph 146, which counts the net increase in local incomes from incremental tourist expenditure. The whole operating cost of the complex is charged.

**7.4 Macroeconomic checks.** The NESDC guide asks large projects to report wider effects, including the balance of payments and labour market effects. These are reported as checks on the results.

**7.5 Sensitivity and risk.** Single-factor cases of 5, 10 and 20 percent on construction cost, operating cost and revenue, a one-year opening delay, two-factor cases of cost and revenue together, and switching values for each key assumption, following ADB (2017) Section VII. The sensitivities are applied to the parameters behind the aggregates, not as mechanical percentage changes.

**7.6 Use of historical evidence.** Published estimates are not transferred to Thailand. The causal benchmark note shows that no causal study of a Disney-type opening exists that could be carried over, and that gross impact studies have no counterfactual. Thai inputs are used wherever they are sourced. Where no Thai value exists, the Hong Kong Disneyland and Ocean Park figures are used as labelled comparables for attendance, visitor mix, ticket price and cost ratios (generalized in the assumptions table). Historical cases inform the ranges and their limits are reported.

**7.7 Reproducibility.** The model is in the project repository (`disney-tourism-thailand/src/dtt/appraisal.py`), with its inputs in `data/appraisal/inputs.csv`. Each row gives its value, unit, source and status. The base case stops only when a sourced or proposal input is pending. The unit tests and repository checks pass.

## 8. Decision rule

The project is supported as a basis for decisions if H3 holds at the 7 percent social discount rate, and if its sign is unchanged under the ±10 percent cost and revenue cases and the one-year delay. Where the sign changes, the result is reported with the conditions under which the project would pass. Break-even attendance is reported for the operator ledger in every case.

## 9. Limitations

- Most Thai demand, spending and operating inputs are not sourced. Several are labelled generalized or assumed values in the assumptions table, and the base case runs on them. Its results are scenario ranges, not sourced estimates.
- AOT traffic for calendar 2024 and for 2019 at Suvarnabhumi and Don Mueang is sourced. U-Tapao has no sourced calendar 2024 or calendar 2019 figure, because AOT does not operate it. Chon Buri occupancy is from National Statistical Office Table 17.1; its 2019 value is unconfirmed. The base occupancy ceiling of 85 percent is an assumption.
- Sources conflict on the high-speed rail opening year (2031 from the EEC Office; Wikipedia gives 2029 in its infobox and 2030 in its own later status entry). The model does not use the rail date. The Chon Buri figure of 28,156,302 is a Thai plus foreign total and is not used as foreign arrivals; the foreign visitor count is 11,689,827.
- The proposal has no signed agreement, approved budget, named operator or site allocation.
- The historical evidence does not identify the Disney-specific effect separately from concurrent policy changes.
- The output is a set of scenario ranges, not confidence intervals.
- The Hong Kong Disneyland and Ocean Park anchors rest on press reports and secondary copies; the operators' primary documents were not read.
## 10. Timeline and responsibilities

| Date | Task | Owner |
| --- | --- | --- |
| 14 Oct | Reframed topic with objectives to the professor | Peem, chow |
| 14 to 15 Oct | Literature sections (a) to (c) | Tai, chow |
| 15 Oct | Research question section | tap |
| 15 Oct | Hypothesis section | chow |
| 15 Oct | Method section | LINK, with Peem |
| 15 Oct | Integration and consistency check | chow, Peem |
| 16 Oct | Submission of the Phase 1 document | Peem |
| After 16 Oct (estimate) | Sourcing of Thai inputs, model build, scenarios, stress tests, final report | Team, Peem |

## 11. References

- Asian Development Bank. (2017). *Guidelines for the economic analysis of projects*. Manila: ADB. https://www.adb.org/sites/default/files/institutional-document/32256/economic-analysis-projects.pdf. DOI: http://dx.doi.org/10.22617/TIM178607-2
- Chan, E. (2004). *Residents' perceptions towards Hong Kong Disneyland and its impacts* [MPhil thesis]. The Hong Kong Polytechnic University. http://hdl.handle.net/10397/85415
- National Economic and Social Development Council. (2024). *Guide to public investment project appraisal* (July 2024 edition). https://www.nesdc.go.th/wp-content/uploads/2025/05/article_20241009124821.pdf
- Plaza, B. (2006). The return on investment of the Guggenheim Museum Bilbao. *International Journal of Urban and Regional Research*, 30(2), 452-467.


## Appendix A. Supplementary track: optimal transport

Scope. This appendix reports the supplementary track. It covers one method, entropic unbalanced optimal transport with an importance-weighted cost, applied to past theme-park openings. The other estimation methods of the earlier analysis are dropped from this appendix. They are the synthetic-control and difference-in-differences effects, the ambient meta-analysis and the ambient fallback of the route rule. The track is supplementary to the bottom-up appraisal in Sections 7 and 8 and has no due date. Its results are diagnostic. They are not an estimate of a Thai effect or of the Thai net benefit.

Method. Optimal transport moves the effects of 19 past openings to the yearly economy features of Thailand. The cost is the squared distance of standardized features, each weighted by its importance from a feature stacking model. The full specification is in the supplement.

Validation. In leave-one-economy-out validation, the importance-weighted transport has an error of 1.015. The equal-weight mean has 1.134 and the uniform-weight transport has 1.006. The importance-weighted transport passes the test against the mean, with a ratio of 0.895. It fails the test against the uniform-weight transport, with a ratio of 1.009 against a limit of 0.95. The importance model is not usable. It has 19 cases against 20 required, a cross-validated R-squared of 0.072 against 0.10, and a permutation p-value of 0.436 against 0.10.

Diagnostic result. For the proposed complex of THB 300 billion, the transported effect is -0.73 percentage points of GDP, with a bootstrap 5 to 95 percent band of -1.28 to -0.19. Because the validation criterion is not met, the band is not evidence of an effect. The result is reported only as a diagnostic.

Later phases. If the team continues this method, the work remaining is to register the weighting choice before use, to extend the sample to at least 20 episodes and repeat the validation, to test the importance model across seeds, and to decide whether the transport enters the appraisal as a cross-check only. The supplement gives the full list and the reproduction steps.
