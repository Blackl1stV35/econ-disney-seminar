"""Compare the stored World Bank indicator files with the World Bank API, cell by cell.

For each of the five indicators, the script reads the four stored files
``data/raw/wdi/wdi_{group}_{indicator}.csv`` and downloads the series of all
economies from the World Bank API (v2, JSON, one request per indicator with
paging).  Only the economies of ``data/raw/wdi/country_metadata.csv`` and the
years that the stored files cover are compared.  Two values agree when they
differ by no more than the relative tolerance (1e-9 by default) of the larger
absolute value; a value that is missing on one side only is reported on its
own.

Run it on a computer with internet access, from the repository root:

    python scripts/verify_wdi_files.py --out checks/wdi

The argument ``--out`` names a directory that receives ``wdi_differences.csv``,
one row for every cell that differs or is present on one side only.  The
summary is printed on the screen.  The command needs only the standard library
and pandas; the sandbox used for the project cannot reach the API.

Exit status: 0 when every compared cell agrees and no cell is present on one
side only; 1 when a cell differs or is present on one side only; 2 when the
check cannot be completed (download, file or argument error).

Options: ``--raw-dir`` (default ``data/raw/wdi``), ``--api-root`` (default
``https://api.worldbank.org/v2``) and ``--tolerance`` (default 1e-9).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

#: Country groups of the stored files; the same labels as ``dtt.panel.GROUPS``.
GROUPS: tuple[str, ...] = ("G1", "G2", "G3", "G4")
#: World Bank indicator codes of the stored files, in the order of ``dtt.panel.INDICATORS``.
INDICATORS: tuple[str, ...] = ("ST.INT.RCPT.CD", "ST.INT.ARVL", "NY.GDP.MKTP.CD", "SP.POP.TOTL", "IS.AIR.PSGR")
METADATA_FILE = "country_metadata.csv"
API_ROOT = "https://api.worldbank.org/v2"
RELATIVE_TOLERANCE = 1e-9
PER_PAGE = 20000
DIFFERENCES_FILE = "wdi_differences.csv"
MISSING_TOKENS = ("", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "..")
COLUMNS = ("indicator", "group", "iso3", "year", "file_value", "api_value", "relative_difference", "status")
STATUS_DIFFERENT = "different"
STATUS_ONLY_FILE = "only in file"
STATUS_ONLY_API = "only in API"

Fetcher = Callable[[str], Any]


def default_raw_dir() -> Path:
    """Return ``data/raw/wdi`` of the repository that contains this script."""
    return Path(__file__).resolve().parents[1] / "data" / "raw" / "wdi"


def download_json(url: str) -> Any:
    """Download a URL and decode its JSON answer; three attempts with a growing pause.

    Raises
    ------
    RuntimeError
        If every attempt fails with a network error.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "dtt-verify-wdi/1.0"})
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after three attempts: {last_error}") from last_error


def _parse_value(text: str, where: str) -> float:
    """Return a stored value; a missing token gives NaN, anything else must be a finite number."""
    stripped = text.strip()
    if stripped in MISSING_TOKENS:
        return math.nan
    try:
        value = float(stripped)
    except ValueError as exc:
        raise ValueError(f"{where}: value {text!r} is not a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where}: value {text!r} is not finite")
    return value


def _parse_year(text: str, where: str) -> int:
    """Return a year that must be a whole number."""
    try:
        value = float(text.strip())
    except ValueError as exc:
        raise ValueError(f"{where}: year {text!r} is not a number") from exc
    if not math.isfinite(value) or value != math.floor(value):
        raise ValueError(f"{where}: year {text!r} is not a whole number")
    return int(value)


def load_economies(path: Path) -> list[str]:
    """Return the economy codes of ``country_metadata.csv``, in file order.

    Raises
    ------
    ValueError
        If the file has no ``iso3`` column or repeats a code.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "iso3" not in frame.columns:
        raise ValueError(f"{path.name}: column iso3 is missing")
    codes = [code.strip() for code in frame["iso3"]]
    repeated = sorted({code for code in codes if codes.count(code) > 1})
    if repeated:
        raise ValueError(f"{path.name}: economy codes repeated: {', '.join(repeated)}")
    return codes


def stored_file_name(group: str, indicator: str) -> str:
    """Return the file name of one stored indicator file, as ``dtt.panel.wdi_filename`` builds it."""
    return f"wdi_{group}_{indicator}.csv"


def load_stored(raw_dir: Path, group: str, indicator: str) -> pd.DataFrame:
    """Read one stored indicator file into the columns iso3, year, value and group.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If a column is missing, a year is not a whole number, a value is not a
        number, or a cell is repeated.
    """
    name = stored_file_name(group, indicator)
    raw = pd.read_csv(raw_dir / name, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    absent = [column for column in ("iso3", "year", "value") if column not in raw.columns]
    if absent:
        raise ValueError(f"{name}: column(s) {absent} missing")
    rows = []
    for index, record in enumerate(raw[["iso3", "year", "value"]].itertuples(index=False), start=2):
        where = f"{name} line {index}"
        rows.append(
            {
                "iso3": str(record.iso3).strip(),
                "year": _parse_year(str(record.year), where),
                "value": _parse_value(str(record.value), where),
                "group": group,
            }
        )
    frame = pd.DataFrame(rows, columns=["iso3", "year", "value", "group"])
    if frame.duplicated(subset=["iso3", "year"]).any():
        raise ValueError(f"{name}: a pair of economy and year is repeated")
    return frame


def api_url(api_root: str, indicator: str, start: int, end: int, page: int) -> str:
    """Return the URL of one page of the series of an indicator for all economies."""
    return (
        f"{api_root.rstrip('/')}/country/all/indicator/{indicator}"
        f"?format=json&per_page={PER_PAGE}&page={page}&date={start}:{end}"
    )


def fetch_series(api_root: str, indicator: str, start: int, end: int, fetch_json: Fetcher) -> pd.DataFrame:
    """Download every page of the series of an indicator and return iso3, year and value.

    Raises
    ------
    RuntimeError
        If an answer of the API does not have the form of a page of records.
    """
    records: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = fetch_json(api_url(api_root, indicator, start, end, page))
        if not (isinstance(payload, list) and len(payload) == 2 and isinstance(payload[0], dict)):
            raise RuntimeError(f"unexpected answer of the API for {indicator}: {str(payload)[:200]}")
        meta, data = payload
        records.extend(data or [])
        if page >= int(meta.get("pages", 1)):
            break
        page += 1
    rows = []
    for record in records:
        iso3 = str(record.get("countryiso3code") or "").strip()
        if not iso3:
            continue
        value = record.get("value")
        rows.append(
            {
                "iso3": iso3,
                "year": _parse_year(str(record["date"]), f"API {indicator}"),
                "value": math.nan if value is None else float(value),
            }
        )
    frame = pd.DataFrame(rows, columns=["iso3", "year", "value"])
    if frame.duplicated(subset=["iso3", "year"]).any():
        raise RuntimeError(f"the API repeats a pair of economy and year for {indicator}")
    return frame


def compare_series(
    indicator: str, stored: pd.DataFrame, api: pd.DataFrame, economies: list[str], tolerance: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compare the stored cells of one indicator with the API cells of the economies of the metadata.

    Returns
    -------
    tuple
        The differing cells, with the columns of ``COLUMNS``, and the counts
        ``compared``, ``equal``, ``different``, ``only_file``, ``only_api`` and
        ``empty_both``.  A cell is compared when both sides hold a value.
    """
    api = api[api["iso3"].isin(economies)]
    merged = stored.merge(api, on=["iso3", "year"], how="outer", suffixes=("_file", "_api"))
    file_value = merged["value_file"].to_numpy(dtype=float)
    api_value = merged["value_api"].to_numpy(dtype=float)
    has_file = ~pd.isna(file_value)
    has_api = ~pd.isna(api_value)
    both = has_file & has_api
    scale = pd.Series(
        [max(abs(a), abs(b)) if ok else math.nan for a, b, ok in zip(file_value, api_value, both)], dtype=float
    ).to_numpy()
    difference = pd.Series([abs(a - b) if ok else math.nan for a, b, ok in zip(file_value, api_value, both)], dtype=float).to_numpy()
    relative = pd.Series(
        [d / s if (ok and s > 0) else (0.0 if ok else math.nan) for d, s, ok in zip(difference, scale, both)], dtype=float
    ).to_numpy()
    equal = both & ((difference <= tolerance * scale) | (difference == 0))
    different = both & ~equal
    only_file = has_file & ~has_api
    only_api = has_api & ~has_file
    counts = {
        "compared": int(both.sum()),
        "equal": int(equal.sum()),
        "different": int(different.sum()),
        "only_file": int(only_file.sum()),
        "only_api": int(only_api.sum()),
        "empty_both": int((~has_file & ~has_api).sum()),
    }
    status = pd.Series(
        [STATUS_DIFFERENT if d else STATUS_ONLY_FILE if f else STATUS_ONLY_API if a else "" for d, f, a in zip(different, only_file, only_api)],
        dtype=object,
    )
    keep = status.to_numpy() != ""
    detail = pd.DataFrame(
        {
            "indicator": indicator,
            "group": merged["group"].fillna("").to_numpy(dtype=object),
            "iso3": merged["iso3"].to_numpy(dtype=object),
            "year": merged["year"].to_numpy(),
            "file_value": file_value,
            "api_value": api_value,
            "relative_difference": relative,
            "status": status.to_numpy(dtype=object),
        }
    )
    return detail.loc[keep, list(COLUMNS)].reset_index(drop=True), counts


def run(
    raw_dir: Path,
    out_dir: Path,
    *,
    fetch_json: Fetcher = download_json,
    api_root: str = API_ROOT,
    tolerance: float = RELATIVE_TOLERANCE,
    stream: Any = None,
) -> int:
    """Compare every stored indicator file with the API, write the differences and print the summary.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Directory with the stored files and ``country_metadata.csv``.
    out_dir : pathlib.Path
        Directory that receives ``wdi_differences.csv``; it is created when absent.
    fetch_json : callable, optional
        Function that takes a URL and returns the decoded JSON answer.
    api_root : str, optional
        Root of the API.
    tolerance : float, optional
        Relative tolerance of the comparison.
    stream : file-like, optional
        Where the summary is printed; standard output by default.

    Returns
    -------
    int
        0 when every compared cell agrees and no cell is present on one side
        only, otherwise 1.
    """
    stream = sys.stdout if stream is None else stream
    economies = load_economies(raw_dir / METADATA_FILE)
    stored_all = {
        indicator: pd.concat([load_stored(raw_dir, group, indicator) for group in GROUPS], ignore_index=True)
        for indicator in INDICATORS
    }
    details = []
    summary: list[dict[str, Any]] = []
    for indicator in INDICATORS:
        stored = stored_all[indicator]
        start, end = int(stored["year"].min()), int(stored["year"].max())
        api = fetch_series(api_root, indicator, start, end, fetch_json)
        detail, counts = compare_series(indicator, stored, api, economies, tolerance)
        details.append(detail)
        summary.append({"indicator": indicator, "years": f"{start}-{end}", **counts})

    out_dir.mkdir(parents=True, exist_ok=True)
    differences = pd.concat(details, ignore_index=True) if details else pd.DataFrame(columns=list(COLUMNS))
    differences.to_csv(out_dir / DIFFERENCES_FILE, index=False)

    header = f"{'indicator':<16}{'years':>11}{'compared':>10}{'equal':>10}{'different':>11}{'only file':>11}{'only API':>10}{'empty':>8}"
    print(f"WDI files against the World Bank API: {len(economies)} economies in {METADATA_FILE}", file=stream)
    print(header, file=stream)
    totals = {"compared": 0, "equal": 0, "different": 0, "only_file": 0, "only_api": 0, "empty_both": 0}
    for row in summary:
        print(
            f"{row['indicator']:<16}{row['years']:>11}{row['compared']:>10}{row['equal']:>10}{row['different']:>11}"
            f"{row['only_file']:>11}{row['only_api']:>10}{row['empty_both']:>8}",
            file=stream,
        )
        for key in totals:
            totals[key] += row[key]
    print(
        f"{'total':<16}{'':>11}{totals['compared']:>10}{totals['equal']:>10}{totals['different']:>11}"
        f"{totals['only_file']:>11}{totals['only_api']:>10}{totals['empty_both']:>8}",
        file=stream,
    )
    print(f"Differing cells written to: {out_dir / DIFFERENCES_FILE}", file=stream)
    bad = totals["different"] + totals["only_file"] + totals["only_api"]
    if bad == 0:
        print("Result: every compared cell agrees with the API.", file=stream)
        return 0
    print(f"Result: {bad} cell(s) differ or are present on one side only.", file=stream)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit status."""
    parser = argparse.ArgumentParser(description="Compare the stored WDI files with the World Bank API, cell by cell.")
    parser.add_argument("--out", required=True, type=Path, help="directory that receives wdi_differences.csv")
    parser.add_argument("--raw-dir", type=Path, default=None, help="directory of the stored files (default data/raw/wdi)")
    parser.add_argument("--api-root", default=API_ROOT, help="root of the API (default %(default)s)")
    parser.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE, help="relative tolerance (default %(default)s)")
    args = parser.parse_args(argv)
    raw_dir = args.raw_dir if args.raw_dir is not None else default_raw_dir()
    try:
        return run(raw_dir, args.out, fetch_json=download_json, api_root=args.api_root, tolerance=args.tolerance)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Check not completed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
