# Phase 1 seminar document

Submission deadline: 16 October 2026. Revised on 19 October 2026 to the finance economics scope.

Public repository: https://github.com/Blackl1stV35/econ-disney-seminar

## 1. Title

Financial and Economic Appraisal of a Proposed Disney-Type Theme-Park Complex in Thailand: A Scenario Analysis.

## 2. Research question and sub-questions

**Primary question.** What are the financial and economic returns of a proposed Disney-type theme-park complex in Thailand at the documented cost of capital and social discount rate, and how sensitive are they to costs, revenue and opening delay?

1. Is the complex financially viable for an operator at the documented cost of capital, and what break-even attendance does it need?
2. Does the complex create positive economic net value for Thailand at the national social discount rate?
3. What are the fiscal flows of public participation?
4. How sensitive are the results to costs, revenue and opening delay?

## 3. Objectives

1. Estimate the operator's financial net present value, internal rate of return and break-even attendance (H1).
2. Estimate the national economic net present value, with economic costs converted at the standard conversion factor (H2).
3. Estimate the fiscal net flows of public participation as a supplementary check (H3).
4. Test the results under the sensitivity and switching cases of section 7.4.

## 4. Hypotheses

- **H1.** The complex has a positive financial net present value under the documented base-case demand and cost assumptions.
- **H2.** The complex produces a positive economic net present value for Thailand at the 7 percent social discount rate.
- **H3 (supplementary).** The discounted incremental tax and fee revenue attributable to the project exceeds its discounted project-specific public expenditure under the stated policy scenario.

**Decision criteria.** H1 holds if the financial net present value is positive at the financial cost of capital (WACC of 8 percent in the base case, recommended; see section 7.3). H2 holds if the economic net present value is positive at the social discount rate of 7 percent (NESDC 2024). H3 holds if the fiscal net present value is positive under the stated policy scenario.

## 5. Scope and scenarios

**Scope.** The topic is finance economics: the financial, economic and fiscal appraisal of an investment. The primary appraisal covers the THB 300 billion sports and entertainment complex, the high-expectation scope of the Thai proposals. The park-only concepts of THB 100 to 200 billion are different projects with their own cost and revenue boundaries, and they are not appraised here.

**Outside the scope.** Tourism demand and receipts as research questions. Visitor numbers, spending and displacement enter only as inputs to the three ledgers. The historical evidence on openings elsewhere is not part of this document.

**Scenarios.** A base case, low and high cases, and the sensitivity cases of section 7.4. The sensitivity cases follow the NESDC (2024) guide: construction cost, operating cost and revenue each varied by 5, 10 and 20 percent, a one-year opening delay, and two-factor cases of cost and revenue together.

## 6. Literature

**(a) Appraisal standards.** The ADB (2017) guidelines set a minimum required economic internal rate of return of 9 percent for investment projects and 6 percent for social sector and environmental projects (paragraph 194), and require sensitivity analysis and switching values (Section VII). The NESDC (2024) guide recommends a social discount rate of 7 percent for appraisals begun in 2025 or later, within a range of 6.55 to 7.59 percent; the rate of 12 percent applies to appraisals begun before 2025. It sets a standard conversion factor of 0.7 for converting financial costs to economic costs where shadow prices cannot be computed.

**(b) Financing and payback benchmark.** Henderson, Chua, Liu and Ngiam (2013) review LEGOLAND Malaysia, the first LEGOLAND in Asia, which opened in Johor in 2012. The article reports that the park's General Manager said full recovery of the investment usually takes 15 to 20 years (Lee, 2012, as cited in the article). This is a press statement, used as a benchmark for the 20-year project life and not as an input. The article also reports a public funding figure that its authors could not verify; that figure is not used.

**(c) Operator comparables.** The attendance, ticket price and cost figures of Hong Kong Disneyland and Ocean Park are used as labelled comparables for attendance, visitor mix, ticket price and cost ratios (generalized in the assumptions table). They are not transfer coefficients.

**Gap.** The literature provides appraisal standards and operator benchmarks, but no finance appraisal of a Thai complex that follows the national guidance. This study fills that gap.

## 7. Method

**7.1 Framework.** A financial and economic appraisal that compares the with-project and without-project situations, following the ADB (2017) guidelines and the NESDC (2024) guide.

**7.2 Inputs and demand.** Visitor numbers and spending are inputs to the ledgers. The model caps incremental visits by the infrastructure that can serve them, namely airport headroom and accommodation supply. Annual passenger movements convert to visits with `visits_from_passengers`, using two movements per round-trip visit, which is an assumption to be confirmed. Incremental visits are capped by headroom in `capped_net_spend`, and net spending follows from the capped visits after displacement.

Sourced values in the input table include U-Tapao terminal capacity of 3.7 million passengers a year (EEC Office, page status October 2025); the AOT six-airport total of 124,033,786 passengers in calendar 2024 and 100,058,695 in fiscal 2023; Suvarnabhumi and Don Mueang traffic for 2019 and 2024 (AOT); Chon Buri occupancy of 80.42 percent in 2024 (National Statistical Office, Table 17.1); Chon Buri foreign visitors of 11,689,827 in 2024 (TAT Intelligence Center); the national arrivals of 2020 to 2023 and of 2025 and the tourism receipts of 2024 and 2025 (Ministry of Tourism and Sports and National Statistical Office); the length of stay of international visitors in 2016, 2017 and 2024 (National Statistical Office and Ministry of Tourism and Sports); the attendance and financial figures of Hong Kong Disneyland and Ocean Park, used as comparators; the Bank of Thailand exchange rate; and the Revenue Department tax rates. Hotel and guesthouse rooms of 956,372 refer to calendar 2021 (NSO Statistical Yearbook Thailand 2024, Table 17.4). Pattaya hotel keys of 31,581 in H2 2024 and 34,743 in H2 2025 are CBRE broker data, not an official source; the 2024 figure is used as a proxy for the room stock of the complex.

All assumptions in the assumptions table were confirmed by the team on 19 October 2026. The team decisions of 14 October 2026 are WACC of 8 percent in the base case (10 and 7 percent in the low and high cases), and a project life of 20 years in the base case (20 and 25 years in the low and high cases). Attendance, incremental and displacement shares, domestic retention, the spending-per-visit base and high cases and the 85 percent occupancy ceiling in the base case are labelled assumptions and are not sourced. The spending-per-visit low case is a sourced on-site benchmark, converted at the Bank of Thailand yen rate of 12 October 2026. Inputs that could not be sourced are blank in the input table, and figures that were not confirmed are flagged unconfirmed, never sourced. The base case stops only when a sourced or proposal input is pending; team decisions do not stop it.

**7.3 Three ledgers.**

- **Operator ledger:** revenue minus operating costs and capital expenditure, giving the financial net present value, internal rate of return and break-even attendance at the financial cost of capital (WACC of 8 percent and a project life of 20 years in the base case, as recommended; the NESDC guide requires both to be stated and justified). Operating costs include depreciation, by the leader's decision.
- **Fiscal ledger:** tax and fee revenue from the net spending minus public capital and operating costs.
- **National economic ledger:** economic benefits minus economic costs, discounted at 7 percent. Economic costs use the conversion factor of 0.7 unless shadow prices are computed. The economic benefit is the net spending of the incremental visits, following ADB (2017) paragraph 146, which counts the net increase in local incomes from incremental expenditure. The whole operating cost of the complex is charged.

**7.4 Sensitivity and risk.** Single-factor cases of 5, 10 and 20 percent on construction cost, operating cost and revenue, a one-year opening delay, two-factor cases of cost and revenue together, and switching values for each key assumption, following ADB (2017) Section VII. The sensitivities are applied to the parameters behind the aggregates, not as mechanical percentage changes.

**7.5 Comparables and reproducibility.** Published tourism-impact estimates are not transferred to Thailand. Where no Thai value exists, the Hong Kong Disneyland and Ocean Park figures are used as labelled comparables (generalized in the assumptions table). The model is in the project repository (`disney-tourism-thailand/src/dtt/appraisal.py`), with its inputs in `data/appraisal/inputs.csv` and its assumptions in `data/appraisal/assumptions.csv`. Each row gives its value, unit, source and status. The unit tests and repository checks pass.

## 8. Decision rule

The project is supported as a basis for decisions if H2 holds at the 7 percent social discount rate, and if the sign of H1 and H2 is unchanged under the ±10 percent cost and revenue cases and the one-year delay. Where a sign changes, the result is reported with the conditions under which the project would pass. Break-even attendance is reported for the operator ledger in every case.

## 9. Limitations

- Most Thai demand, spending and operating inputs are not sourced. Several are labelled generalized or assumed values in the assumptions table, and the base case runs on them. Its results are scenario ranges, not sourced estimates.
- Demand inputs such as airport headroom and accommodation supply come from official sources where sourced, and otherwise from labelled proxies (section 7.2).
- The proposal has no signed agreement, approved budget, named operator or site allocation.
- The output is a set of scenario ranges, not confidence intervals.
- Several comparator anchors rest on secondary copies: the Ocean Park attendance for FY2018/19 (a search summary of the results announcement), the Lotte World ticket price for 2026 (a news report) and the Shanghai Disneyland attendance for 2019 (a copy of an index). The Ocean Park operating costs for FY2018/19 come from its annual report.

## 10. References

- Asian Development Bank. (2017). *Guidelines for the economic analysis of projects*. Manila: ADB. https://www.adb.org/sites/default/files/institutional-document/32256/economic-analysis-projects.pdf. DOI: http://dx.doi.org/10.22617/TIM178607-2
- Henderson, J. C., Chua, P. T., Liu, H. J., & Ngiam, J. (2013). Theme parks in Southeast Asia: A case study of LEGOLAND Malaysia. *Asia Pacific Journal of Innovation in Hospitality and Tourism*, 2(2), 227-242.
- Lee, J. (2012). LEGOLAND Malaysia ready to receive visitors. *The Star Online*, 13 August. (As cited in Henderson et al., 2013.)
- National Economic and Social Development Council. (2024). *Guide to public investment project appraisal* (July 2024 edition). https://www.nesdc.go.th/wp-content/uploads/2025/05/article_20241009124821.pdf
