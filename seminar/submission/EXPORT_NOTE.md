# Export note: submission files

Exported: 20 October 2026. Font: Times New Roman for text, Courier New for code. Format: A4, printed from the HTML of the markdown source (Python markdown module, then Microsoft Edge headless print to PDF). Edge's own print header and footer (date, title and local file path) were removed afterwards with PyMuPDF, so the printed pages carry no browser text.

| File | Pages | Bytes | SHA-256 | Source | Fonts embedded |
|---|---|---|---|---|---|
| Phase1_seminar_document_2026-10-20.pdf | 4 | 151469 | `68f43e5344ba278799aa70139651df39cef5fe6d6273f6f06c699b7f01ebd7e1` | seminar/PHASE1_DRAFT.md as reframed on 20 October 2026 (Financial Appraisal only) | TimesNewRomanPS-BoldMT, TimesNewRomanPSMT, TimesNewRomanPS-ItalicMT, CourierNewPSMT |
| Supplement_closure_record_2026-10-16.pdf | 12 | 330610 | `6cc8f8e1d837b8de60624de33ad6fe67aa5ce024b38294622a77c301a0cf2c8d` | seminar/SUPPLEMENT_CLOSURE_RECORD.md | AAAAAA+TimesNewRomanPS-BoldMT, BAAAAA+TimesNewRomanPSMT, CAAAAA+CourierNewPSMT |

## Status of this export

- The Phase 1 PDF was re-exported on 20 October 2026 from the source reframed to Financial Appraisal only. Its title is "Scenario-Based Financial Feasibility and Break-Even Analysis of a Proposed Disney-Type Theme-Park Complex in Thailand". It replaces the 19 October PDF, which was never merged to main and is removed from this branch; the history keeps it.
- The earlier 16 October file (16 pages, with Appendix A) stays in git history.
- The economic hypothesis H2 and the fiscal hypothesis H3 of the 19 October source are not in this document. The national NPV is not reported as a conclusion.
- Checked for this export: page count (4), page size (A4), fonts (Times New Roman and Courier New only), and no em or en dashes in the PDF text. Not checked: rendering in Microsoft Word.
- Not re-exported in this change: the supplement PDF. Its source, SUPPLEMENT_CLOSURE_RECORD.md, changed after its export, so the supplement PDF stays out of date until its re-export is assigned.
- Final results document: Final_results_2026-10-19.pdf (5 pages, 7576627 bytes, SHA-256 `9f822f6223291b2a71ad301a3a06b02e00faee1ceade47ec4d3745c59f65a927`), built from seminar/FINAL_RESULTS_2026-10-19.md with its three figures in seminar/figures/. It records the 19 October economic and financial results and does not replace the Phase 1 PDF.
- The export is replaced, and this note is updated, whenever the Phase 1 text changes.
