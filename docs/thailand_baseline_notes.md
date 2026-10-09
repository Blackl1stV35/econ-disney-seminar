# Thailand baseline series and theme-park proposal status: data notes

Prepared 2026-10-08. Companion to `data/processed/thailand_baseline.csv`. Part 1 documents the baseline series for 2015 to 2025. Part 2 records the status of theme-park and Disney-type proposals for Thailand as of 2026-10-08. Data vintages: World Bank WDI of 2026-07-13, Bank of Thailand (BoT) tables to August 2026, Ministry of Tourism and Sports (MoTS) releases to June 2026, IMF Country Report No. 26/41 of February 2026.

## Part 1. Baseline series, 2015 to 2025

### 1.1 File conventions

The CSV (148 rows) is in long format with the header `year,variable,value,unit,source,url,note`. Variables without a prefix are the adopted series: seven variables with eleven annual values each. Variables with the prefix `alt_` hold alternative definitions, earlier data vintages and cross-checks. They are not part of the baseline and are kept so that the choice of series can be tested. A cell with several URLs separates them with a vertical bar. Each note begins with ADOPTED or ALTERNATIVE and records the revision status where it is known. All monetary values are at current prices. Baht values of receipts are derived from US$ values (section 1.4).

| Variable | Content | Years |
|---|---|---|
| `intl_tourism_receipts_usd_bn`, `intl_tourism_receipts_thb_bn` | Balance-of-payments travel credits (adopted) | 2015 to 2025 |
| `intl_arrivals_million` | International tourist arrivals, MoTS series (adopted) | 2015 to 2025 |
| `gdp_current_usd_bn`, `gdp_current_thb_bn`, `fx_thb_per_usd` | GDP at current prices and average exchange rate, WDI (adopted) | 2015 to 2025 |
| `receipts_pct_gdp` | Adopted receipts in baht over GDP in baht (adopted) | 2015 to 2025 |
| `alt_receipts_mots_*` | MoTS revenue from international tourists: baht, US$ at the WDI rate, percent of GDP, first-published headline values, implied current value for 2024 | 2015 to 2019 and 2025; first-published 2020, 2023, 2024 |
| `alt_receipts_bot_*` | BoT tourism receipt (narrow definition), with the earlier vintage for 2025 (table released 2026-01-30) | 2024, 2025 |
| `alt_receipts_wdi_*` | WDI total receipts (ST.INT.RCPT.CD), travel items (ST.INT.TVLR.CD) and passenger transport items (ST.INT.TRNR.CD) | 2015 to 2020 |
| `alt_receipts_imf_usd_bn` | Travel credits in IMF Country Report No. 26/41 | 2020 to 2025 |
| `alt_arrivals_*` | WDI arrivals, first-published MoTS arrivals, BoT arrivals | 2015 to 2019; 2016, 2022, 2023; 2024, 2025 |
| `alt_gdp_current_thb_bn_imf`, `alt_fx_bot_thb_per_usd` | IMF nominal GDP in baht; BoT average exchange rate | 2020 to 2025; 2021 to 2025 |

### 1.2 Adopted series and reasons

| Variable | Adopted series | Reason |
|---|---|---|
| International tourism receipts (US$ and THB) | Balance-of-payments travel credits compiled by the BoT, read from WDI as the travel share of service exports (BX.GSR.TRVL.ZS) times service exports (BX.GSR.NFSV.CD, BoP, current US$). Baht values are the US$ values times the annual average rate. | One compiler, one definition and full coverage for 2015 to 2025. Identical to WDI travel-items receipts (ST.INT.TVLR.CD) in 2015 to 2019 and within US$ 0.05 bn of the IMF figures (rounded to 0.1) for 2020 to 2024. Excludes passenger transport, so it is narrower than WDI ST.INT.RCPT.CD (section 1.4). |
| International arrivals | MoTS count of international tourist arrivals: revised series for 2015 to 2019 (equal to WDI), National Statistical Office (NSO) Statistical Yearbook 2025 for 2020 to 2024, MoTS preliminary count for 2025. | Official count; WDI stops after 2019; revised values are preferred to first-published values. |
| GDP (US$ and THB) | WDI NY.GDP.MKTP.CD and NY.GDP.MKTP.CN, based on national accounts compiled by the National Economic and Social Development Council (NESDC). | Same vintage in both currencies and consistent with the exchange rate. Within 0.6 percent of the IMF vintage of February 2026 in every year from 2020 to 2025. |
| Average THB per US$ | WDI PA.NUS.FCRF, official rate, period average. | Reconciles WDI GDP in baht to GDP in US$ within rounding. BoT averages for 2021 to 2025 differ by at most 0.04 baht. |
| Receipts as percent of GDP | Adopted receipts in baht divided by GDP in baht. | Both inputs share the same basis. The ratio is computed, not published. |

### 1.3 Adopted values

| Year | Receipts, US$ bn | Receipts, THB bn | Arrivals, million | GDP, US$ bn | GDP, THB bn | THB per US$ | Receipts, % of GDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 41.25 | 1,412.6 | 29.923 | 401.3 | 13,743.5 | 34.25 | 10.28 |
| 2016 | 44.79 | 1,580.8 | 32.530 | 413.4 | 14,590.3 | 35.30 | 10.83 |
| 2017 | 52.38 | 1,777.6 | 35.592 | 456.4 | 15,488.7 | 33.94 | 11.48 |
| 2018 | 56.37 | 1,821.2 | 38.178 | 506.8 | 16,373.3 | 32.31 | 11.12 |
| 2019 | 59.81 | 1,857.0 | 39.916 | 544.0 | 16,889.2 | 31.05 | 11.00 |
| 2020 | 12.53 | 392.0 | 6.702 | 500.3 | 15,655.4 | 31.29 | 2.50 |
| 2021 | 4.35 | 139.2 | 0.428 | 506.1 | 16,182.0 | 31.98 | 0.86 |
| 2022 | 14.70 | 515.5 | 11.065 | 495.7 | 17,379.6 | 35.06 | 2.97 |
| 2023 | 30.65 | 1,066.7 | 28.150 | 517.0 | 17,993.2 | 34.80 | 5.93 |
| 2024 | 42.28 | 1,492.1 | 35.546 | 529.4 | 18,683.9 | 35.29 | 7.99 |
| 2025 (P) | 44.77 | 1,472.3 | 32.974 | 577.0 | 18,973.7 | 32.88 | 7.76 |

P: preliminary. Full-precision values and the URL for every value are in the CSV. The status of each 2023 to 2025 value is in section 1.7.

### 1.4 Receipts: definitions and the choice among them

Four measures of spending by international visitors are carried in the CSV, and a fifth is noted for reference. They differ in coverage and in revision behaviour.

Balance-of-payments travel credits (adopted). The BoT compiles the balance of payments from the International Transactions Reporting System, MoTS travel data, surveys, reports from other agencies and a data model (BoT table description EC_XT_046). The travel line excludes international passenger transport, which is recorded under transport. WDI publishes the same concept as ST.INT.TVLR.CD, and the values agree with the adopted series for 2015 to 2019.

BoT tourism receipt (alternative). The quarterly press-release tables carry a baht series named Tourism Receipt. Its footnote states that it excludes health, education, excursionists and purchases by short-term workers, which implies that the broader travel credits include them. The series equals 0.929 of the adopted series in 2024 and 0.931 in 2025. It appears only for 2024 and 2025 in the documents retrieved.

MoTS revenue from international tourists (alternative). The ministry publishes a baht estimate of revenue from international tourists. It exceeds the adopted series by 3.0 to 3.3 percent in 2015 to 2019, by 4.4 percent in 2025 (preliminary) and by about 8 percent in 2024, using the current-vintage 2024 value implied by the reported 2025 decline of 4.71 percent. Headline figures at first publication were higher than later values: 2016 was first published at 1,641.3 and later stands at 1,633.5, and 2024 was first published at about 1,670 against the implied 1,612.5. The 2024 press report gives the 1.67 trillion beside domestic revenue of THB 950 billion and its headline attributes the 1.67 trillion to foreign visitors. The ministry's total tourism revenue for 2025 (THB 2,703,335 million) adds domestic travel to the international component and is not comparable with receipts.

WDI international tourism receipts (alternative, ST.INT.RCPT.CD). The series comes from UN Tourism and includes payments to national carriers. For Thailand it equals travel items plus passenger transport items (ST.INT.TVLR.CD plus ST.INT.TRNR.CD) in each year from 2015 to 2020. Passenger transport was 7.1 to 8.2 percent of the total. The vintage of 2026-07-13 has no value after 2020.

Tourism Satellite Account (reference only, not in the CSV). The MoTS Tourism Satellite Account report for fiscal year 2017 gives inbound tourism expenditure of THB 1,494.9 billion for 2015 (revised) and THB 1,694.0 billion for 2016 (provisional), which are 5.8 and 7.2 percent above the adopted series.

| Measure | 2019 | 2024 | 2025 | Ratio to adopted series |
|---|---:|---:|---:|---|
| Balance-of-payments travel credits (adopted), THB bn | 1,857.0 | 1,492.1 | 1,472.3 (P) | 1.000 |
| Bank of Thailand tourism receipt (narrow), THB bn | not retrieved | 1,386.3 | 1,370.3 (P) | 0.929 in 2024; 0.931 in 2025 |
| Ministry revenue from international tourists, THB bn | 1,911.8 | 1,612.5 (implied current value; first published about 1,670) | 1,536.6 (P) | 1.030 in 2019; 1.081 in 2024; 1.044 in 2025 |
| WDI ST.INT.RCPT.CD, US$ bn | 64.37 | none | none | 1.076 in 2019 (RCPT over travel items) |

Revision and timing. The BoT incorporates MoTS travel data in an annual revision at the end of September that covers the two preceding years (EC_XT_046). The end-September 2026 revision of 2024 and 2025 therefore post-dates the WDI vintage used here and pre-dates these notes. The BoT database was not re-queried, so the adopted values for 2024 and 2025 may since have changed. Separately, the BoT moved its balance-of-payments publication to the BPM7 classification with effect from 2026-06-30: 22 tables were replaced by new series with history from 2005, and the services classification grew from 10 to 15 categories. The notice does not mention travel and does not say whether the history of the old series was revised. Whether the WDI vintage reflects any BPM7 restatement was not established.

Baht values. Baht receipts are the US$ values times the WDI average rate. The BoT converts balance-of-payments dollar flows to baht at monthly average mid-rates, so its own baht values can differ slightly. A test on the narrow tourism receipt (quarterly baht receipts divided by quarterly average rates, summed in dollars and converted at the annual average rate) gives a total 0.2 percent above the published baht total in 2024 and 0.2 percent below it in 2025, which indicates the order of magnitude of the difference.

Bridge to WDI ST.INT.RCPT.CD. Analysis that uses ST.INT.RCPT.CD for other economies needs Thailand on the same basis. In 2015 to 2019 the ratio of that series to travel items was 1.076 to 1.089. Passenger transport credits after 2020 were not available, so a published like-for-like figure does not exist for 2021 to 2025. Scaling the adopted series by the 2015 to 2019 ratio gives roughly US$ 45.5 to 46.1 billion for 2024 and US$ 48.2 to 48.8 billion for 2025. These are approximations and are not included in the CSV.

### 1.5 Arrivals

The adopted series is the MoTS count of international tourist arrivals. For 2015 to 2019 the revised MoTS series equals the WDI values (29.923, 32.530, 35.592, 38.178 and 39.916 million). For 2020 to 2024 the values come from Tables 17.1 and 17.3 of the NSO Statistical Yearbook 2025, which give 2020 and 2021 in thousands to two decimals (6,702.40 and 427.87) and later years to the person. A footnote in those tables states that the counts exclude Thai nationals residing abroad. The 2025 value of 32,974,321 is the MoTS preliminary count, and the BoT table of August 2026 shows the same total (32,974 thousand). Arrivals fell 7.23 percent from 2024 to 2025 and stood 17.4 percent below the 2019 level.

First-published and revised figures differ. The 2016 preliminary count was 32,588,303 against a revised 32,529,588. The 2022 total was first reported as 11.15 million against 11,065,226 in the NSO table. The 2023 total was first reported as 28,042,131 against 28,150,016. The causes of the 2022 and 2023 differences were not established. For 2021 the NSO table gives 427.87 thousand. A secondary compilation (https://en.wikipedia.org/wiki/Tourism_in_Thailand) lists 819,429 in its yearly table and 427,869 in its country-breakdown total, and the NSO figure is adopted.

### 1.6 GDP and exchange rate

GDP and the exchange rate are taken from WDI (NY.GDP.MKTP.CD, NY.GDP.MKTP.CN and PA.NUS.FCRF, last updated 2026-07-13). GDP in baht divided by the average rate reproduces GDP in US$ within rounding in every year. IMF Country Report No. 26/41 gives nominal GDP in baht for 2020 to 2025. Relative to WDI the IMF values differ by +0.04, +0.03, -0.01, -0.21, -0.54 and -0.05 percent in 2020 to 2025. The IMF report is older than the WDI vintage, and its 2025 value is a staff estimate. BoT average rates for 2021 to 2025 are 32.00, 35.06, 34.81, 35.26 and 32.87 baht per US$. NESDC announced real growth of 2.4 percent for 2025 on 2026-02-16, which is the first annual estimate. NESDC tables could not be read directly because the site disallows automated retrieval.

### 1.7 Status of the 2023 to 2025 figures

Final means that no scheduled revision was identified. Revised means revised at least once and open to further change. Preliminary means a first annual estimate. Estimate means a staff or forecast value.

| Variable | 2023 | 2024 | 2025 |
|---|---|---|---|
| Receipts, BoP travel credits (adopted) | Final after annual revisions | Revised once (September 2025 cycle); a further revision was due at the end of September 2026 | Preliminary. IMF February 2026 staff estimate: US$ 43.8 bn |
| Receipts, BoT tourism receipt (alternative) | Not retrieved | As published; unchanged between the January 2026 and August 2026 tables | Preliminary; revised from 1,376.6 (January 2026 table) to 1,370.3 (August 2026 table) |
| Receipts, MoTS revenue (alternative) | First-published headline only (about THB 1,200 bn) | Implied current value 1,612.5; first published about 1,670 | Preliminary: 1,536.6 |
| Arrivals (adopted) | Final | Final | Preliminary |
| GDP, US$ and THB (adopted) | Final after revision; WDI is 0.21 percent above the IMF vintage | Revised; WDI is 0.54 percent above the IMF vintage, so further change is possible | Preliminary first annual estimate. IMF staff estimate was THB 18,963.6 bn |
| Average THB per US$ (adopted) | Final | Final | Final |
| Receipts as percent of GDP (adopted) | Final | Revised | Preliminary |

### 1.8 Source conflicts

1. Receipts level in 2025 (THB bn): the BoT tourism receipt is 1,370.3, balance-of-payments travel credits are 1,472.3 and MoTS revenue is 1,536.6. The highest is 12.1 percent above the lowest. The differences are definitional (section 1.4).
2. Direction of the 2025 change: MoTS reports a decline of 4.71 percent in baht. Travel credits fall 1.3 percent in baht but rise 5.9 percent in US$, because the average rate moved from 35.29 to 32.88 baht per US$. The BoT tourism receipt falls 1.2 percent in baht. Arrivals fell 7.2 percent, so receipts per arrival in US$ rise from 1,189 to 1,358 on the adopted series. This is a further reason to treat the 2025 figure as preliminary.
3. Dollar conversion of the MoTS figure: the ministry states US$ 48.8 billion for 2025 international revenue, which implies 31.5 baht per US$ against the BoT average of 32.87. At the average rate used in the CSV the same baht value is US$ 46.7 billion.
4. First-published versus revised ministry figures differ for revenue (2016, 2024) and arrivals (2016, 2022, 2023), as set out in sections 1.4 and 1.5.
5. Receipts in 2020: WDI travel items give US$ 14.198 bn, the current balance-of-payments vintage gives 12.527 bn and the IMF gives 12.5 bn. The adopted series uses the balance-of-payments vintage.
6. GDP: IMF and WDI differ by up to 0.54 percent (section 1.6). The adopted series uses WDI.
7. The BoT tourism receipt for 2025 was revised from 1,376.6 to 1,370.3 between the January and August 2026 tables.
8. Arrivals for 2021: NSO 427.87 thousand against 819,429 in one secondary table (section 1.5).

### 1.9 Limitations

The MoTS spreadsheets for arrivals and revenue could not be read, so MoTS values come from MoTS PDF reports, NSO yearbook tables and press reports of ministry releases. MoTS revenue for 2021 and 2022 was not retrieved, and the 2020, 2023 and 2024 values are first-published headline figures rounded by the press. NESDC pages could not be read, so GDP rests on WDI with the IMF cross-check. The BoT's own travel-credit series in baht under the BPM7 classification, and the end-September 2026 revision, were not retrieved. Passenger transport credits after 2020 are not available. Domestic tourism revenue is outside the scope of the file.

### 1.10 Principal sources

- World Bank WDI API for Thailand (pattern `https://api.worldbank.org/v2/country/THA/indicator/<code>?format=json&date=2015:2025&per_page=100`): BX.GSR.TRVL.ZS, BX.GSR.NFSV.CD, ST.INT.TVLR.CD, ST.INT.TRNR.CD, ST.INT.RCPT.CD, ST.INT.ARVL, NY.GDP.MKTP.CD, NY.GDP.MKTP.CN, PA.NUS.FCRF. Indicator definitions: https://api.worldbank.org/v2/indicator/ST.INT.RCPT.CD?format=json and https://api.worldbank.org/v2/indicator/ST.INT.TVLR.CD?format=json
- BoT balance-of-payments table description (EC_XT_046): https://app.bot.or.th/BTWS_STAT/statistics/DownloadFile.aspx?file=EC_XT_046_ENG.PDF
- BoT notice on the BPM7 transition: https://www.bot.or.th/content/dam/bot/documents/th/statistics/BOP_Jun2026.pdf
- BoT quarterly press-release tables: https://www.bot.or.th/content/dam/bot/documents/en/thai-economy/state-of-thai-economy/monthly-report/macro-2026-08-table_q.pdf and the same path with macro-2025-12-table_q.pdf (attached to BoT Press Release No. 4/2026 of 2026-01-30, https://www.bot.or.th/en/news-and-media/news/news-20260130.html), macro-2024-12-table_q.pdf and macro-2022-12-table_q.pdf
- MoTS reports: https://www.mots.go.th/download/article/article_20170216111201.pdf, https://www.mots.go.th/download/article/article_20190819124714.pdf, https://www.mots.go.th/download/article/article_20201104090605.pdf; Tourism Satellite Account report: https://www.mots.go.th/download/article/article_20190206181451.pdf; 2025 file listings: https://www.mots.go.th/news/category/806 and https://www.mots.go.th/news/category/836
- NSO Statistical Yearbook 2025, Tables 17.1 and 17.3: https://www.nso.go.th/public/e-book/Statistical-Yearbook/SYB-2025_webPage/File%20SYB/2025/17/T%2017.1%202025.pdf and the same path with T%2017.3%202025.pdf
- 2025 year-end ministry figures: https://thailand.prd.go.th/en/content/category/detail/id/2078/iid/466167, https://asianews.network/thailand-tourism-slips-in-2025-despite-domestic-growth/ and https://www.pattayamail.com/news/thailands-tourism-sees-revenue-dip-as-domestic-travel-shows-resilience-532748
- First-published figures: https://www.pattayamail.com/thailandnews/thailands-tourism-booms-in-2024-with-over-35-54-million-foreign-visitors-generating-1-67-trillion-baht-in-revenue-486768 (2024), https://bangkokpost.com/thailand/general/2716846/tourist-arrivals-top-28m-in-2023 (2023), https://www.pattayamail.com/travel/thailand-tourism-ends-2020-down-74-no-revival-seen-until-2022-338692 (2020), https://www.bangkokpost.com/business/general/2590979 (2022 arrivals)
- IMF Country Report No. 26/41: https://www.imf.org/-/media/files/publications/cr/2026/english/1thaea2026001-source-pdf.pdf
- NESDC 2025 growth announcement, as reported by the Government Public Relations Department: https://thailand.prd.go.th/en/content/category/detail/id/2078/iid/477029

## Part 2. Theme-park and Disney-type proposals, status as of 2026-10-08

### 2.1 Summary

No signed commitment for a Disney-scale or other mega theme park in Thailand was found in the sources reviewed, which run through 2026-10-07. The "Disneyland Thailand" idea is a government proposal advanced since December 2025 by Deputy Prime Minister and Transport Minister Phiphat Ratchakitprakarn, who oversees the Eastern Economic Corridor (EEC) and chaired the EEC Policy Committee meeting of 2026-06-05, with the EEC Office (EECO). It replaced the earlier casino-based Entertainment Complex plan, whose bill was removed from the parliamentary agenda on 2025-07-07. The most recent concept describes about THB 300 billion of investment in EEC Capital City (EECiti), Huay Yai subdistrict, Bang Lamung district, Chonburi: a theme park of about 3,000 rai (480 hectares) costing THB 100 to 200 billion, and a sports and entertainment complex with an 80,000-seat stadium. No confirmation of interest from the Walt Disney Company was found. On 2026-03-27 the minister stated that no negotiations had been held with it, and a decision on whether to proceed was said to be expected by the end of 2026.

The only government approval found relates to infrastructure at the same site. On 2026-06-05 the EEC Policy Committee approved a public-private partnership (PPP) framework for EECiti infrastructure and utilities, about THB 72.04 billion of private investment under a 50-year build-operate-transfer concession. The package does not include a theme park. EECO held a market sounding for it on 2026-07-21. Two EECiti agreements published in October 2026, a memorandum with Dongnan International Group (H.K.) Limited signed on 2026-09-10 and a collaboration with UOB Thailand announced on 2026-10-07, concern investor promotion and do not mention a theme park.

The latest sources reviewed are EECO releases dated through 2026-10-07, which contain no theme-park item, and a Thailand Business News overview dated 2026-08-26, which still describes the theme park as a floated idea. The latest dated report on the initiative itself is a Leisure Opportunities article of 2026-05-15 summarising a Royal Thai Embassy statement, and the latest ministerial statement with figures is dated 2026-03-27. EECi (Eastern Economic Corridor of Innovation, Wang Chan Valley, Rayong, https://www.eeci.or.th/) is a different project from EECiti, and none of the sources reviewed links the theme park to EECi.

Next dated milestones reported: summary of the market-sounding feedback to the EEC Policy Committee by October 2026 and submission to the Cabinet within 2026 (Bangkok Biznews, 2026-07-22); results of the entertainment hub study in November, year not stated (Pattaya Mail, 2026-06-07); decision on the Disney initiative by the end of 2026 (Nation, 2026-03-27).

### 2.2 Status by attribute

| Attribute | Status | Latest dated source |
|---|---|---|
| Sponsor | Thai government. Lead: Phiphat Ratchakitprakarn. Implementing office: EECO (Secretary-General Chula Sukmanop). The Sports Authority of Thailand is involved for sports facilities. No private sponsor is named. An unnamed Abu Dhabi investor and unnamed Thai investors are reported as interested. | 2026-03-27; 2026-02-17 |
| Location | EECiti (2,339 ha, about 14,619 rai), Huay Yai, Bang Lamung, Chonburi. Early statements (2025-12-09) allowed four EEC provinces. | 2026-08-26 |
| Budget | About THB 300 bn in total, of which the park is nearly THB 200 bn (2026-02-12) or THB 100 to 200 bn (2026-03-27), subject to feasibility. No approved budget. | 2026-03-27 |
| Planned size | Park up to 3,000 rai (480 ha). With the sports and entertainment complex, about 800 ha (5,000 rai). Stadium of 80,000 seats. | 2026-03-27; 2026-02-12 |
| Visitor forecast | Ministerial claim of 10 million additional visitors a year, revenue above THB 150 bn, over 100,000 jobs and about 1 percent of GDP a year if realised. No study is cited. | 2026-02-04 |
| Timeline | Decision on the Disney initiative by end-2026; completion within four years of a start. EECiti PPP: Cabinet in 2026, investor invitation in 2027 or 2028 (sources differ), construction from 2028. Sports complex feasibility was due in August 2026 and the entertainment hub study in November. | 2026-03-27; 2026-08-26 |
| Operator contact | Plan to invite the Walt Disney Company to invest directly, with licensing as a fallback (2026-01-12). The only reported contact is a congratulatory letter to Disney's new chief executive (2026-03-27). The statement of 2026-05-15 reports no indication that the company is involved. | 2026-05-15; 2026-03-27 |
| Commitment | Proposal. No signed agreement, approved budget, named operator or site allocation for the theme park. Approved: the EECiti infrastructure PPP framework, without the park (2026-06-05). | 2026-10-07 |

### 2.3 Dated evidence

| Date | Source and sponsor | Content | Status | URL |
|---|---|---|---|---|
| 2025-07-07 | Coalition whip Visuth Chainaroon, via iGaming Business | Entertainment Complex Bill (five integrated resorts with gaming) removed from the parliamentary agenda | Shelved | https://igamingbusiness.com/casino/thailand-casino-bill-shelved/ |
| 2025-12-09 | Phiphat Ratchakitprakarn, via The Nation and Thaiger (2025-12-10) | "Magnet projects" for the high-speed rail to U-Tapao: a world-class amusement park (Disneyland a possible operator) on no more than 3,000 rai (4.8 million square metres) and an 80,000-seat stadium, Pattaya-Chonburi area; EEC Office to study feasibility; no budget or timeline; may not finish within the current government's term | Proposal; preliminary study | https://www.nationthailand.com/news/tourism/40059480 ; https://thethaiger.com/news/national/thai-government-studies-disneyland-project-instead-of-casino-complex |
| 2026-01-12, 2026-01-27 | Phiphat, via Khaosod English and The Nation | Plan to invite the Walt Disney Company to invest directly, with a licensed developer as fallback; project described as feasible and under PPP study; no casino; continuation depends on the next government | Proposal under study | https://www.khaosodenglish.com/life/2026/01/12/thailand-eyes-southeast-asias-first-disneyland/ ; https://www.nationthailand.com/news/policy/40061763 |
| 2026-02-03, 2026-02-04 | Phiphat; Royal Thai Embassy in Washington | Embassy positions Thailand as an "important option" for the first Disneyland in Southeast Asia. Claims: 10 million additional visitors a year, revenue above THB 150 bn, over 100,000 jobs, about 1 percent of GDP a year, if realised. No study cited | Ministerial claims; no study cited | https://thestandard.co/pipit-disneyland-thailand-eec-economy/ ; https://www.thansettakij.com/economy/megaproject/650582 ; https://www.nst.com.my/amp/world/region/2026/02/1370869/all-ears-thailand-courts-disneyland-major-tourism-play |
| 2026-02-12 | The Nation (blog), attributing to Phiphat | About THB 300 bn in Chonburi under the EEC: theme park nearly THB 200 bn on about 480 ha; sports and entertainment centre over THB 100 bn on about 320 ha. No timeline; next steps are feasibility studies and licensing structures. Disney has not confirmed; idea raised by at least five governments in 26 years | Proposal | https://www.nationthailand.com/blogs/business/investment/40062428 |
| 2026-02-17 to 2026-02-19 | EECO Secretary-General Chula Sukmanop; Phiphat | Park could sit in EECiti (over 15,000 rai, about 5,000 rai for sports and entertainment). Government to seek direct Disney investment or a licensed development with private partners; Thai investors reported interested | Proposal | https://thailand.go.th/issue-focus-detail/thailand-eyes-disneyland-style-theme-park-in-eec-to-boost-mega-projects ; https://www.kaohooninternational.com/economics/577292 ; https://www.thansettakij.com/economy/megaproject/651853 |
| 2026-03-27 | Phiphat, via The Nation | Interest from an unnamed Abu Dhabi investor. Park of 3,000 rai inside EECiti (about 15,000 rai); THB 100 to 200 bn for the park, about THB 300 bn with the sports complex, subject to feasibility. Decision by end-2026; completion within four years if started. Government and EEC "have not yet held negotiations" with Disney | Proposal; no negotiation | https://www.nationthailand.com/business/investment/40064324 |
| 2026-05-15 | Royal Thai Embassy statement, via Leisure Opportunities | Embassy: government courting global theme-park operators. Phiphat quoted: work under way to identify a site and to develop incentives for Disney or another major company. THB 300 bn complex (480 ha park, 80,000-seat stadium) to be delivered as a PPP with private capital. The report notes no indication that the Walt Disney Company is involved | Proposal; no operator | https://www.leisureopportunities.co.uk/news/Thai-government-reveals-it-is-courting-major-theme-park-operators-to-develop-a-major-attraction-near-Bangkok/362939 |
| 2026-06-05 (reported 06-06, 06-07) | EEC Policy Committee chaired by Phiphat | Approved the PPP framework for EECiti infrastructure and utilities: about THB 72.04 bn private investment, 50-year build-operate-transfer, 6,168 rai ready; boundary changes need Cabinet reconsideration. Sports complex feasibility due August 2026; entertainment hub study due November. Theme park not in the package | Approved (infrastructure only) | https://www.nationthailand.com/business/40067106 ; https://www.pattayamail.com/news/new-sports-entertainment-and-smart-city-hub-planned-for-eec-552491 |
| 2026-07-21 (reported 07-22, 08-05) | EECO market sounding | More than 200 attendees (Bangkok Biznews) or more than 100 (TR Journal). Summary to the EEC committee by October 2026, Cabinet within 2026, investor invitation in 2027 (Bangkok Biznews) or partner selection in 2028 (TR Journal), construction 2028. No Disney or theme-park mention; no investor selected | Consultation | https://www.bangkokbiznews.com/economics/1244130 ; https://www.trjournalnews.com/85723/ |
| 2026-08-26 | Thailand Business News overview | EECiti 2,339 ha in Huay Yai; sports and entertainment complex about 240 ha; theme park "floated" with no formal approval and no Disney partner named; PPP bidding early 2028 | Concept only | https://www.thailand-business-news.com/investment/324421-thailands-eastern-economic-corridor-capital-city-eeciti-key-developments-and-investment-opportunities |
| 2026-09-10 to 2026-10-07 | EECO | Memorandum with Dongnan International Group (H.K.) Limited signed 2026-09-10 (published 2026-10-06, no value stated). Collaboration with UOB Thailand announced 2026-10-07; EECiti described as a THB 1.1 trillion mixed-use business city without defining the figure. Neither mentions a theme park. The EECO news listing from 2026-08-20 to 2026-10-07 has no theme-park item | Signed or announced, none for the park | https://www.eeco.or.th/en/press_release/eeco-mou-dongman-en/ ; https://www.eeco.or.th/en/press_release/eeco-and-uob-en/ ; https://www.eeco.or.th/en/news |

### 2.4 Scale checks against the baseline

These calculations place the ministerial claims next to the adopted baseline. They are arithmetic checks, not forecasts.

| Check | Calculation | Result |
|---|---|---|
| Claimed 10 million additional visitors against arrivals | 10 / 32.974; 10 / 35.546; 10 / 39.916 | 30.3 percent of 2025 arrivals; 28.1 percent of 2024; 25.1 percent of the 2019 peak |
| Claimed visitors against a Disney comparator | 10 / 14.7 (Shanghai Disneyland, 2024 attendance of 14.7 million, TEA Global Experience Index, reported 2025-10-27 at https://english.shanghai.gov.cn/en-Latest-WhatsNew/20251027/63097f5dd6ff4d2dad7e736573ead8d6.html) | 68 percent of that park's total attendance, which includes domestic visitors |
| Claimed THB 150 bn revenue against receipts | 150 / 1,472.3; 150 / 1,536.6; 150 / 18,973.7 | 10.2 percent of 2025 travel credits; 9.8 percent of 2025 MoTS revenue; 0.79 percent of 2025 GDP |
| Implied revenue per additional visitor | THB 150 bn over 10 million | THB 15,000, or 33.6 percent of 2025 travel credits per arrival (THB 44,649) and 32.2 percent of MoTS revenue per arrival (THB 46,599) |
| Claimed GDP effect | 1 percent of 2025 GDP | THB 189.7 bn a year, above the THB 150 bn revenue claim |
| Capital cost | THB 300 bn | US$ 9.1 bn at the 2025 average rate; 1.58 percent of 2025 GDP |

The sources do not state whether the THB 150 billion refers to park revenue or total visitor spending, whether the 10 million visitors are all visitors or foreign visitors only (one outlet says foreign), or whether the 1 percent GDP effect is a direct or total effect. For reference, arrivals fell by 2.57 million between 2024 and 2025.

### 2.5 Existing large parks and other private projects

The chair of the International Association of Amusement Parks and Attractions board, who is also managing director of Siam Park Bangkok, and the association's chief executive said on 2024-05-29 that Thailand has seen no mega theme-park investment beyond water parks and small hotel or mall parks, and that foreign operators need changes to laws, tax rules and infrastructure (https://www.nationthailand.com/news/tourism/40038403). Attendance at Thailand's existing parks was not retrieved.

Recent private attraction projects are smaller than the Disney-type proposal and are not Disney-type parks. Asset World's Jurassic-themed attraction at Asiatique in Bangkok received Board of Investment approval for THB 1.2 billion on 4,000 square metres, with no opening date stated and no publication date shown on the page retrieved (https://www.bangkokpost.com/thailand/general/2958265/tourism-push-to-go-prehistoric-with-new-jurassic-park). ICONPHUKET (Siam Piwat, Thalang district, Phuket) was unveiled on 2026-10-05 at THB 10 billion with opening planned for 2029, within the wider Synthesis Ark Phuket project (CV Group, THB 50 billion, over 500 rai, phase 1 in 2032). The report cites a forecast of 80,000 to 100,000 visitors a day with no stated basis; sustained over a year that range equals 29.2 to 36.5 million visits, similar in size to Thailand's total international arrivals. One photo caption in the report gives 2019 as the opening year, which conflicts with 2029 in the text (https://thethaiger.com/news/phuket/phuket-gets-new-10-billion-baht-destination-iconphuket).

### 2.6 Conflicts in the proposal record

1. Budget: THB 300 billion in total with the park at nearly THB 200 billion (Nation, 2026-02-12; Leisure Opportunities, 2026-05-15), the park alone at THB 100 to 200 billion (Nation, 2026-03-27), and up to US$ 10 billion without a stated scope (Thai Examiner, 2026-02-11, https://www.thaiexaminer.com/thai-news-foreigners/2026/02/11/disneyland-plan-to-take-its-first-steps-forward-this-week-as-transport-ministry-holds-exploratory-meeting/).
2. Size: a park of up to 3,000 rai (480 ha) in December 2025 and March 2026; 144 to 480 ha cited from studies (thailand.go.th, 2026-02-17); about 800 ha in total as 480 ha plus 320 ha (Nation, 2026-02-12); about 240 ha for the sports and entertainment complex (Thailand Business News, 2026-08-26); about 5,000 rai requested for the stadium and theme park (Thansettakij, 2026-02-23, https://www.thansettakij.com/economy/megaproject/652202).
3. EECiti PPP timeline: invitation or bidding in early 2027 (Nation, 2026-06-06; Pattaya Mail, 2026-06-07; Bangkok Biznews, 2026-07-22) against partner selection or bidding in 2028 (TR Journal, 2026-08-05; Thailand Business News, 2026-08-26). The later sources may reflect a slip from the 2027 target, but no official statement reconciling the two was found.
4. Market-sounding attendance: more than 200 (Bangkok Biznews) against more than 100 (TR Journal; Thailand Business News).
5. EECiti private investment: THB 72.04 billion in all sources, with an earlier estimate of about THB 74.4 billion mentioned by Thailand Business News. EECO describes EECiti itself as a THB 1.1 trillion city on 2026-10-07 without defining the figure, so the PPP value, the theme-park concept value and the EECO figure refer to different scopes.
6. Operator: Disney is named as the target of incentives and of a direct-investment invitation, but sources of 2026-03-27 and 2026-05-15 report no negotiation and no indication that the company is involved. An unnamed Abu Dhabi investor is the only reported expression of interest.
