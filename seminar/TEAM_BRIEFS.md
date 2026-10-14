# Team briefs (14 October 2026)

Direction: the leader, Peem, has locked the seminar direction. The leader's decisions are final. Each brief states the deliverable, the corrections to the current group drafts, and the check before the item is accepted. Section numbers refer to `PHASE1_DRAFT.md`.

Locked direction: Appraising a Proposed Disney-type Theme-Park Complex in Thailand, bottom-up. Primary scope: the THB 300 billion complex. Objectives and hypotheses H1 to H3 are approved; H4 is supplementary. Working hypothesis: public infrastructure variation is the main driver of the tourism effect, so incremental visits are capped by infrastructure headroom. Historical evidence is calibration and context only.

## Corrections to the current group drafts (apply to every section)

1. One primary research question. Remove the alternative framings ("Tourism Revenue Focus" and "Mega Project Appraisal").
2. Remove the econometric response text, the US$1.01 billion summary and the "recommended estimate" language.
3. Do not present any historical estimate as a Thai result, a causal effect of Disney or a net benefit.
4. Use the approved wording for H1 to H3. H4 is labelled supplementary.
5. One title, one scope and one set of roles.
6. Citations must be complete. Verified: Chan (2004), PolyU repository, handle 10397/85415. Removed from the references until confirmed: Tan, How and Koh (Universal Studios Singapore). Milman (2010) has a single author according to the publisher.

## Briefs by role

### tap: research questions (due 15 October)
- Deliverable: section 2, one primary question and four sub-questions, each answerable by the appraisal.
- Check: each sub-question names the output that answers it (incremental spending, operator NPV and break-even, economic NPV, fiscal flows and sensitivity).

### chow: hypotheses and integration (15 October)
- Deliverable: section 4 (H1 to H3, H4 supplementary) with the decision criteria, then the merged document and the consistency check above.
- Check: every objective maps to a hypothesis, and every criterion names a model output.

### Tai: literature (15 October)
- Deliverable: section 6, groups (a) to (e), with complete citations. Include the Singapore review requested on 13 October.
- Check: every source is in section 11 with a full record, or is left out. News sources are used for context only. The 2025 arrivals and receipts figures need a Tourism Authority of Thailand or Ministry of Tourism and Sports source before they carry a number.
- Open: confirm the Tan, How and Koh record in the NTU repository (handle 10356/15092) in a browser. The record could not be fetched automatically.

### LINK: research methods (15 October)
- Deliverable: section 7, the bottom-up method with the infrastructure cap, the three ledgers, scenarios, sensitivity cases and macro checks.
- Check: every parameter is listed as sourced or pending in `data/appraisal/inputs.csv`. Use the NESDC (2024) values (social discount rate 7 percent, conversion factor 0.7, sensitivity cases of 5, 10 and 20 percent, cost, O&M and revenue, and a one-year delay) and the ADB (2017) sensitivity and switching-value rules.
- Correction: the econometric synthetic control and difference-in-differences designs are the supplementary track, not the primary method.

### Peem: integration, method sign-off, model and submission (Phase 1 due 16 October)
- Deliverable: sign-off of sections 2 to 9, the submission on 16 October, and the model in `src/dtt/appraisal.py`, which already implements the capacity cap and the ledgers.
- Decisions taken as leader: (1) infrastructure variation is the main driver; (2) Thai inputs come from public infrastructure and tourism sources first, then from established working-model inputs, and otherwise the primary result is revised to what the seminar can ground; (3) a missing input is shown as a labelled assumption, never as a sourced value.
- Supplementary track (no due date): the earlier econometric items (outcome choice, donor pool, synthetic DiD, concurrent shocks), used only if they succeed.

## Thai inputs (status on 14 October 2026)

- Sourced: 17 values, each with its source, page or quote. Secondary sources are labelled as such.
- Pending: 56 values, including calendar 2024 and 2019 airport traffic, Chon Buri room and occupancy data, spending per visit, incremental share, displacement, attendance, project life and the financial cost of capital.
- Established working-model inputs were not found in any source fetched so far. If the leader names one, it enters the table as a borrowed input with its source.
- Still open: a fetchable URL for the NESDC (2024) Guide. The copy used is the downloaded PDF.
