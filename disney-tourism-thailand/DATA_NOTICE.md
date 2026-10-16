# Data notice

This notice covers the data files of the repository. It does not cover the program code in `src/`, `tests/`, `scripts/` and `notebooks/`.

## World Bank indicators

The files `data/raw/wdi/wdi_G1_*.csv` to `data/raw/wdi/wdi_G4_*.csv` hold World Development Indicators (WDI) published by the World Bank. The World Bank makes these data available under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, and under its open data terms, https://datacatalog.worldbank.org/public-licenses.

- Source: World Development Indicators, https://datatopics.worldbank.org/world-development-indicators/ (API: https://api.worldbank.org/v2/).
- Vintage of the files: 2026-07-13.
- Indicators, one file per country group and indicator:
  - `ST.INT.RCPT.CD`: international tourism, receipts (current US$);
  - `ST.INT.ARVL`: international tourism, number of arrivals;
  - `NY.GDP.MKTP.CD`: GDP (current US$);
  - `SP.POP.TOTL`: population, total;
  - `IS.AIR.PSGR`: air transport, passengers carried.
- Economies: the 96 economies listed in `data/raw/wdi/country_metadata.csv`, in four groups (G1 to G4).

The values in the files are stored as the World Bank publishes them, without rescaling or other change. The panel construction in `src/dtt/panel.py` transforms them: it builds the outcome (international tourism receipts as a percentage of GDP) and the other columns, and notebook 00 writes the result to `data/processed/global_panel.csv`.

Attribution for re-use: World Bank, World Development Indicators, accessed through the API and the files of the vintage above.

## Single-country panel

The file `data/raw/disney_did_panel.csv` is the panel of nine economies used by the single-country analysis (`src/dtt/replication.py`, `src/dtt/did.py` and notebook 01). It is the data set of the original seminar analysis and was supplied by the project owner. This repository grants no licence for this file. Its origin is to be stated by the project owner before the repository is made public.
