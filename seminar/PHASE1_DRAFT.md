# Phase 1 seminar document

Submission deadline: 16 October 2026. Revised on 19 October 2026 to the finance economics scope, on 20 October 2026 to Financial Appraisal only, and on 21 October 2026 to the park-only operator boundary with cash tax and replacement capex (workbook v2), with ticket plus on-site income as the primary revenue.

Public repository: https://github.com/Blackl1stV35/econ-disney-seminar

## 1. Title

Scenario-Based Financial Feasibility and Break-Even Analysis of a Proposed Disney-Type Theme Park in Thailand.

## 2. Research question and sub-questions

**Primary question.** Is a proposed Disney-type theme park in Thailand financially viable for an operator at the documented cost of capital, what break-even attendance and break-even capital cost does it need, and how sensitive is that result to costs, revenue and opening delay?

1. Is the theme park financially viable for an operator at the documented cost of capital (WACC of 8 percent in the base case)?
2. What break-even attendance and break-even capital cost does the theme park need?
3. How sensitive are the operator results to construction cost, operating cost, revenue and opening delay?

## 3. Objectives

1. Estimate the operator's financial net present value, internal rate of return, break-even attendance and break-even capital cost (H1).
2. Test the results under the sensitivity and switching cases of section 7.4.
3. Show the depreciation treatment explicitly: depreciation is excluded from cash flows, and the earlier confirmed basis is kept as a reconciliation (section 7.3).

## 4. Hypothesis

- **H1.** The theme park has a positive financial net present value for the operator under the documented base-case demand and cost assumptions.

**Decision criterion.** H1 holds if the financial net present value after tax is positive at the financial cost of capital (WACC of 8 percent in the base case, recommended; see section 7.3). The economic and fiscal hypotheses of the earlier draft are retired from this document. They are not tested here, and no national welfare or public-finance conclusion is drawn.

## 5. Scope and scenarios

**Scope.** The topic is financial appraisal of the operator's park-only ledger. The capital boundary is the park-only proposal of about THB 200 billion (the high park scenario of the Thai proposals, reported in the news and not verified by an official statement). It is matched to the theme-park revenue model, so that capital and revenue cover the same asset. The THB 300 billion sports and entertainment complex is not the primary boundary: its capital covers assets whose revenue this model does not model. It is kept as a reconciliation to the earlier model and reported separately (section 7.3).

**Outside the scope.** The national economic appraisal (national benefit, economic costs and the conversion factor), the fiscal flows of public participation, and tourism demand and receipts as research questions. Visitor numbers, spending and displacement enter only as inputs to the operator ledger. The historical evidence on openings elsewhere is not part of this document. The sports and entertainment complex is not appraised as a separate asset.

**Scenarios.** A base case, low and high cases, and the sensitivity cases of section 7.4. The sensitivity cases follow the NESDC (2024) guide: construction cost, operating cost and revenue each varied by 5, 10 and 20 percent, a one-year opening delay, and two-factor cases of cost and revenue together.

## 6. Literature

**(a) Appraisal standards.** The ADB (2017) guidelines require sensitivity analysis and switching values (Section VII), and this study applies both to the operator ledger. The ADB economic rate test and the NESDC (2024) social discount rate are not applied in this frame, and neither is the NESDC standard conversion factor.

**(b) Financing and payback benchmark.** Henderson, Chua, Liu and Ngiam (2013) review LEGOLAND Malaysia, the first LEGOLAND in Asia, which opened in Johor in 2012. The article reports that the park's General Manager said full recovery of the investment usually takes 15 to 20 years (Lee, 2012, as cited in the article). This is a press statement, used as a benchmark for the 20-year project life and not as an input. The article also reports a public funding figure that its authors could not verify; that figure is not used.

**(c) Operator comparables.** The attendance, ticket price and cost figures of Hong Kong Disneyland and Ocean Park are used as labelled comparables for attendance, visitor mix, ticket price and cost ratios (generalized in the assumptions table). They are not transfer coefficients.

**Gap.** The literature provides operator benchmarks and appraisal guidance, but no operator-level financial appraisal of a Thai theme park that tests break-even attendance and break-even capital cost under sensitivity cases. This study fills that gap.

## 7. Method

**7.1 Framework.** An operator-level financial appraisal of the park-only after-tax cash flows, discounted at the financial cost of capital, with the sensitivity and switching tests of ADB (2017) Section VII applied to the financial result.

**7.2 Inputs and demand.** Visitor numbers and spending are inputs to the operator ledger. The model caps incremental visits by the infrastructure that can serve them, namely airport headroom and accommodation supply. Annual passenger movements convert to visits with `visits_from_passengers`, using two movements per round-trip visit, which is an assumption to be confirmed. Incremental visits are capped by headroom in `capped_net_spend`, and net spending follows from the capped visits after displacement. The operator visits are the attendance less the international visits above the physical cap, which is the smaller of the airport capacity and the room stock at full occupancy.

Sourced values in the input table include U-Tapao terminal capacity of 3.7 million passengers a year (EEC Office, page status October 2025); the AOT six-airport total of 124,033,786 passengers in calendar 2024 and 100,058,695 in fiscal 2023; Suvarnabhumi and Don Mueang traffic for 2019 and 2024 (AOT); Chon Buri occupancy of 80.42 percent in 2024 (National Statistical Office, Table 17.1); Chon Buri foreign visitors of 11,689,827 in 2024 (TAT Intelligence Center); the national arrivals of 2020 to 2023 and of 2025 and the tourism receipts of 2024 and 2025 (Ministry of Tourism and Sports and National Statistical Office); the length of stay of international visitors in 2016, 2017 and 2024 (National Statistical Office and Ministry of Tourism and Sports); the attendance and financial figures of Hong Kong Disneyland and Ocean Park, used as comparators; the Bank of Thailand exchange rate; the standard corporate income tax rate of 20 percent; and the Revenue Department tax rates. Hotel and guesthouse rooms of 956,372 refer to calendar 2021 (NSO Statistical Yearbook Thailand 2024, Table 17.4). Pattaya hotel keys of 31,581 in H2 2024 and 34,743 in H2 2025 are CBRE broker data, not an official source; the 2024 figure is used as a proxy for the room stock of the complex.

**Attendance benchmark.** Break-even attendance is set against the largest parks that are reported.

| Park | Attendance (million) | Year | Source |
|---|---|---|---|
| Magic Kingdom (Walt Disney World) | 17.8 | 2024 | TEA/AECOM ranking as reported by Disney Food Blog (23 October 2025) |
| Disneyland Park (California) | 17.3 | 2024 | TEA/AECOM ranking as summarised by BGT Safari (11 November 2025) |
| Hong Kong Disneyland | 7.5 | FY2025 | Hong Kong Disneyland press release (28 April 2026), as recorded in the input table |
| Shanghai Disneyland (approximate) | 14 | 2023 | TEA/AECOM 2023 index, approximate; not checked against the report |
| Tokyo Disney Resort (two parks) | not verified | | Not verified in this round |

The primary break-even attendance of 43 million visits a year is about 2.4 times the Magic Kingdom figure.

All assumptions in the assumptions table were confirmed by the team on 19 October 2026. The team decisions of 14 October 2026 are WACC of 8 percent in the base case (10 and 7 percent in the low and high cases), and a project life of 20 years in the base case (20 and 25 years in the low and high cases). Attendance, incremental and displacement shares, domestic retention, the spending-per-visit base and high cases and the 85 percent occupancy ceiling in the base case are labelled assumptions and are not sourced. The spending-per-visit low case is a sourced on-site benchmark, converted at the Bank of Thailand yen rate of 12 October 2026. Inputs that could not be sourced are blank in the input table, and figures that were not confirmed are flagged unconfirmed, never sourced.

**7.3 Operator ledger.** The park-only ledger has revenue less cash operating costs, capital expenditure, replacement capital expenditure and corporate income tax. It gives the financial net present value, internal rate of return, break-even capital cost and break-even attendance at the financial cost of capital (WACC of 8 percent and a project life of 20 years in the base case, as recommended; the NESDC guide requires both to be stated and justified). The elements are set as follows:

- **Operating costs.** Cash basis: 0.7712 of revenue, the Hong Kong Disneyland FY2025 cost before depreciation and amortisation as a share of revenue. Depreciation is excluded from cash flows, and capital is charged only through capital expenditure.
- **Capital.** The park-only boundary of about THB 200 billion (proposal, not verified), spent in the two construction years.
- **Replacement capital expenditure.** No source is in the repository, so the base case is zero. A stress case uses the depreciation share of the Hong Kong Disneyland noncurrent assets (about 7.9 percent a year) as a proxy. It is a depreciation rate, not a replacement estimate. Validation on 21 October 2026 finds the proxy not appropriate as a base: it is depreciation across all noncurrent assets, including long-lived buildings and land. At 200 bn it is about 15.8 bn a year, around two thirds of primary revenue, so it is used only as a labelled stress.
- **Corporate income tax.** 20 percent on taxable income, with straight-line tax depreciation of the capital over the 20-year tourism asset life from opening. Losses are carried forward without a time limit, which is a proxy for the five-year Thai rule. Cumulative taxable income is negative over the life in the base case, so the tax is zero and the rule has no effect on the result.

**Revenue.** Ticket revenue at 2,877 THB per operator visit, plus on-site income at 1,862 THB per operator visit (the sourced low-case benchmark). Kamolpeen decided on 21 October 2026 that on-site income is part of the primary revenue, with the same weight as team approval.

Primary result (provisional, pre-tax, 8 percent): park-only NPV -144.05 bn; break-even capital cost 50.4 bn; break-even attendance 43 million visits a year, about 2.4 times the Magic Kingdom attendance. The after-tax result equals the pre-tax result, because taxable income is negative in every operating year.

The earlier confirmed basis (operating costs at 0.9383 of revenue, which includes depreciation, with capital charged again) and the 300 billion complex capital are kept as a reconciliation column. The reconciliation reproduces the earlier cash-basis result of the legacy model, so the change of boundary and cost basis is traceable. The depreciation overlap is decided by kamolpeen on 21 October 2026, with the same weight as team approval. The primary result remains subject to the open items in section 9.

**7.4 Sensitivity and risk.** Single-factor cases of 5, 10 and 20 percent on construction cost, operating cost and revenue, a one-year opening delay, two-factor cases of cost and revenue together, and a replacement-capex proxy case, following ADB (2017) Section VII. The sensitivities are applied to the parameters behind the aggregates, not as mechanical percentage changes.

**7.5 Comparables and reproducibility.** Published tourism-impact estimates are not transferred to Thailand. Where no Thai value exists, the Hong Kong Disneyland and Ocean Park figures are used as labelled comparables (generalized in the assumptions table). The model is in the project repository (`disney-tourism-thailand/src/dtt/appraisal.py`), with its inputs in `data/appraisal/inputs.csv` and its assumptions in `data/appraisal/assumptions.csv`. Each row gives its value, unit, source and status. The financial workbook (`seminar/workbook/Phase1_financial_appraisal_v2.xlsx`) recalculates the park-only ledger from the same inputs with live formulas. The unit tests in the repository cover the appraisal code; they were not rerun for this reframing.

## 8. Decision rule

The project is supported as a basis for decisions if H1 holds at the financial cost of capital after tax and the sign of H1 is unchanged under the ±10 percent cost and revenue cases and the one-year delay. Where the sign changes, the result is reported with the conditions under which the project would pass. Break-even attendance and break-even capital cost are reported in every case.

## 9. Limitations

- Most Thai demand, spending and operating inputs are not sourced. Several are labelled generalized or assumed values in the assumptions table, and the base case runs on them. Its results are scenario ranges, not sourced estimates.
- Demand inputs such as airport headroom and accommodation supply come from official sources where sourced, and otherwise from labelled proxies (section 7.2).
- The proposal has no signed agreement, approved budget, named operator or site allocation. The park-only capital of about THB 200 billion is a news figure, not an official statement.
- The output is a set of scenario ranges, not confidence intervals.
- The replacement capital expenditure is unsourced and zero in the base case. The tax loss rule is a proxy. Working capital and the public-funding flows are not in the operator ledger.
- The depreciation overlap is decided by kamolpeen (21 October 2026), with the same weight as team approval. The confirmed basis, which charges depreciation inside operating costs and the capital separately, is kept as a reconciliation only.
- The primary result includes on-site income at the sourced 1,862 THB benchmark per operator visit. The ticket-only result (-163.12 bn before tax) is kept as a memo (workbook scenario table, row 104).
- Several comparator anchors rest on secondary copies: the Ocean Park attendance for FY2018/19 (a search summary of the results announcement), the Lotte World ticket price for 2026 (a news report) and the Shanghai Disneyland attendance for 2019 (a copy of an index). The Ocean Park operating costs for FY2018/19 come from its annual report.
- The economic and fiscal ledgers of the earlier draft are not reported here. No national welfare conclusion is drawn.

## 10. References

- Disney Food Blog (23 October 2025). Magic Kingdom is the most visited theme park in the world (TEA/AECOM ranking). https://disneyfoodblog.com/2025/10/23/magic-kingdom-is-the-1-most-visited-theme-park-in-the-world-but-the-rest-of-the-list-may-surprise-you
- BGT Safari (11 November 2025). The TEA rankings are out for the top 25 theme parks worldwide. https://bgtsafari.com/2025/11/11/the-tea-rankings-are-out-of-for-the-top-25-theme-parks-worldwide-and-almost-half-are-in-asia/
- Asian Development Bank. (2017). *Guidelines for the economic analysis of projects*. Manila: ADB. https://www.adb.org/sites/default/files/institutional-document/32256/economic-analysis-projects.pdf. DOI: http://dx.doi.org/10.22617/TIM178607-2
- Henderson, J. C., Chua, P. T., Liu, H. J., & Ngiam, J. (2013). Theme parks in Southeast Asia: A case study of LEGOLAND Malaysia. *Asia Pacific Journal of Innovation in Hospitality and Tourism*, 2(2), 227-242.
- Lee, J. (2012). LEGOLAND Malaysia ready to receive visitors. *The Star Online*, 13 August. (As cited in Henderson et al., 2013.)
- National Economic and Social Development Council. (2024). *Guide to public investment project appraisal* (July 2024 edition). https://www.nesdc.go.th/wp-content/uploads/2025/05/article_20241009124821.pdf
