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

**Decision criteria.** H1 holds if the base-case net incremental spending is positive after displacement. H2 holds if the financial net present value is positive at the financial cost of capital (PENDING: WACC). H3 holds if the economic net present value is positive at the social discount rate of 7 percent (NESDC 2024). H4 holds if the fiscal net present value is positive under the stated policy scenario.

## 5. Scope and scenarios

**Scope.** The primary appraisal covers the THB 300 billion sports and entertainment complex, the high-expectation scope of the Thai proposals. The park-only concepts of THB 100 to 200 billion are different projects with their own demand, revenue and cost boundaries. They are reported as separate cases only if time allows.

**Scenarios.** A base case, low and high demand cases, and the sensitivity cases of section 7.5. The sensitivity cases follow the NESDC (2024) guide: construction cost, operating cost and revenue each varied by 5, 10 and 20 percent, a one-year opening delay, and two-factor cases of cost and revenue together.

## 6. Literature

**(a) Appraisal standards.** The ADB (2017) guidelines set a minimum required economic internal rate of return of 9 percent for investment projects and 6 percent for social sector and environmental projects (paragraph 194), and require sensitivity analysis and switching values (Section VII). The NESDC (2024) guide recommends a social discount rate of 7 percent for appraisals begun in 2025 or later, within a range of 6.55 to 7.59 percent; the rate of 12 percent applies to appraisals begun before 2025. It sets a standard conversion factor of 0.7 for converting financial costs to economic costs where shadow prices cannot be computed.

**(b) Theme-park and single-attraction evidence.** Milman (2010) reviews the global theme-park industry and treats the sector as dependent on economic and regulatory conditions (PENDING: confirm against the full text). Plaza (2006) estimates the effect of the Guggenheim Museum Bilbao on overnight stays, jobs and the return on investment, which makes it the standard single-attraction case.

**(c) Economic impact and cost-benefit analysis.** Burgan and Mules (2001) argue that economic impact and cost-benefit analysis measure different things and can be reconciled only when resources are idle or underused. Dwyer, Forsyth, Madden and Spurr (2000) show that tourism growth displaces other activity, and recommend computable general equilibrium models for the economy-wide effects.

**(d) Historical cases.** Hong Kong Disneyland and Shanghai Disney Resort are the closest cases. Their evidence is used for calibration ranges and as context, not as a transfer coefficient (section 7.6). Chan (2004) studies residents' perceptions of Hong Kong Disneyland (PENDING: citation details). Tan, How and Koh study Universal Studios Singapore (PENDING: year and citation details).

**(e) Causal evaluation methods (historical track only).** Abadie (2021) and Cameron and Miller (2015) provide the guidance for synthetic control and cluster-robust inference used in the supplementary historical analysis.

**Gap.** The literature provides single-attraction, macroeconomic and appraisal-standard evidence, but no bottom-up model for a Thai complex that follows the national appraisal guidance. This study fills that gap.

## 7. Method

**7.1 Framework.** A bottom-up appraisal that compares the with-project and without-project situations, following the ADB (2017) guidelines and the NESDC (2024) guide.

**7.2 Demand and spending.** Visits are split into segments: Thai residents, international short-haul, international long-haul and repeat visitors. For each segment the team estimates attendance, the incremental share, length of stay, spending per visit, the opening ramp-up and capacity limits (all PENDING). Displaced spending at other Thai destinations is subtracted.

**7.3 Three ledgers.**
- **Operator ledger:** revenue minus operating costs and capital expenditure, giving the financial net present value, internal rate of return and break-even attendance at the financial cost of capital (PENDING: WACC and project life, which the NESDC guide requires to be stated and justified).
- **Fiscal ledger:** tax and fee revenue from the net spending minus public capital and operating costs.
- **National economic ledger:** economic benefits minus economic costs, discounted at 7 percent. Economic costs use the conversion factor of 0.7 unless shadow prices are computed. The economic benefit definition must be confirmed against ADB (2017) before use (PENDING).

**7.4 Macroeconomic checks.** The NESDC guide asks large projects to report wider effects, including the balance of payments and labour market effects. These are reported as checks on the results.

**7.5 Sensitivity and risk.** Single-factor cases of 5, 10 and 20 percent on construction cost, operating cost and revenue, a one-year opening delay, two-factor cases of cost and revenue together, and switching values for each key assumption, following ADB (2017) Section VII. The sensitivities are applied to the parameters behind the aggregates, not as mechanical percentage changes.

**7.6 Use of historical evidence.** A transfer of historical estimates from Hong Kong and Shanghai to Thailand was tested in a separate step and did not give a stable result. The primary estimate therefore uses Thai inputs only. Historical cases inform the ranges of attendance and spending, and their limits are reported.

**7.7 Reproducibility.** The model is in the project repository (`disney-tourism-thailand/src/dtt/appraisal.py`). Its inputs are in `data/appraisal/inputs.csv`, where each row states its source and status. The model refuses to run while an input is pending.

## 8. Decision rule

The project is supported as a basis for decisions if H3 holds at the 7 percent social discount rate, and if its sign is unchanged under the ±10 percent cost and revenue cases and the one-year delay. Where the sign changes, the result is reported with the conditions under which the project would pass. Break-even attendance is reported for the operator ledger in every case.

## 9. Limitations

- The Thai inputs (demand, spending, operating costs, tax rates, financial cost of capital and project life) are not yet sourced. No government document was found that states the assumptions behind the official projection of Thai GDP.
- The proposal has no signed agreement, approved budget, named operator or site allocation.
- The historical evidence does not identify the Disney-specific effect separately from concurrent policy changes.
- The output is a set of scenario ranges, not confidence intervals.

## 10. Timeline and responsibilities

| Date | Task | Owner |
| --- | --- | --- |
| 14 Oct | Reframed topic with objectives to the professor | Peem, chow |
| 14 to 15 Oct | Literature sections (a) to (e) | Tai, chow |
| 15 Oct | Research question section | tap |
| 15 Oct | Hypothesis section | chow |
| 15 Oct | Method section | LINK, with Peem |
| 15 Oct | Integration and consistency check | chow, Peem |
| 16 Oct | Submission of the Phase 1 document | Peem |
| After 16 Oct (estimate) | Sourcing of Thai inputs, model build, scenarios, stress tests, final report | Team, Peem |

## 11. References

- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391-425. https://doi.org/10.1257/JEL.20191450
- Asian Development Bank. (2017). *Guidelines for the economic analysis of projects*. Manila: ADB. https://www.adb.org/sites/default/files/institutional-document/32256/economic-analysis-projects.pdf
- Burgan, B., & Mules, T. (2001). Reconciling cost-benefit and economic impact assessment for event tourism. *Tourism Economics*, 7(4), 321-330. https://doi.org/10.5367/000000001101297892
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372. https://doi.org/10.3368/jhr.50.2.317
- Chan, E. (2004). *Residents' perceptions towards Hong Kong Disneyland and its impacts* [MPhil thesis]. The Hong Kong Polytechnic University. (PENDING: check the citation against the library record.)
- Dwyer, L., Forsyth, P., Madden, J. R., & Spurr, R. (2000). Economic impacts of inbound tourism under different assumptions regarding the macroeconomy. *Current Issues in Tourism*, 3(4), 325-363. https://doi.org/10.1080/13683500008667877
- Milman, A. (2010). The global theme park industry. *Worldwide Hospitality and Tourism Themes*, 2(3), 220-237. https://doi.org/10.1108/17554211011052177
- National Economic and Social Development Council. (2024). *Guide to public investment project appraisal* (July 2024 edition). https://www.nesdc.go.th/wp-content/uploads/2025/05/article_20241009124821.pdf
- Plaza, B. (2006). The return on investment of the Guggenheim Museum Bilbao. *International Journal of Urban and Regional Research*, 30(2), 452-467. https://doi.org/10.1111/j.1468-2427.2006.00672.x
- Tan, W., How, S., & Koh, S. (PENDING: year and title details). Universal Studios Singapore: A case study.


