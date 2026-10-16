"""Tests of scripts/verify_wdi_files.py with a fake fetcher; no test uses the network."""

from __future__ import annotations

import importlib.util
import io
import math
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

from dtt import panel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_wdi_files", ROOT / "scripts" / "verify_wdi_files.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

ECONOMIES = ["AAA", "BBB", "CCC"]
GROUP_OF = {"AAA": "G1", "BBB": "G2", "CCC": "G3"}
YEARS = [2000, 2001, 2002]


def _world() -> dict[str, dict[tuple[str, int], float]]:
    """Deterministic values of the five indicators for the three economies and three years."""
    world = {}
    for position, indicator in enumerate(verify.INDICATORS):
        cells = {}
        for e, iso3 in enumerate(ECONOMIES):
            for t, year in enumerate(YEARS):
                cells[(iso3, year)] = (1.0 + position) * 1000.0 + 37.0 * e + 1.0 / 3.0 * (t + 1)
        world[indicator] = cells
    return world


def _write_raw(raw_dir: Path, stored: dict[str, dict[tuple[str, int], float]]) -> None:
    """Write the metadata file and the twenty indicator files in the layout of the repository."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"iso3": ECONOMIES, "name": ["A", "B", "C"], "region": ["r"] * 3, "income_level": ["i"] * 3}
    ).to_csv(raw_dir / "country_metadata.csv", index=False)
    for group in verify.GROUPS:
        for indicator in verify.INDICATORS:
            rows = [
                {"iso3": iso3, "year": year, "value": "" if value is None else repr(value)}
                for (iso3, year), value in sorted(stored[indicator].items())
                if GROUP_OF.get(iso3) == group
            ]
            frame = pd.DataFrame(rows, columns=["iso3", "year", "value"])
            frame.to_csv(raw_dir / verify.stored_file_name(group, indicator), index=False)


class FakeAPI:
    """Stands in for the World Bank API: answers the URLs of ``api_url`` from a dictionary of cells."""

    def __init__(self, cells: dict[str, dict[tuple[str, int], float | None]], page_size: int | None = None):
        self.cells = cells
        self.page_size = page_size
        self.urls: list[str] = []

    def __call__(self, url: str):
        self.urls.append(url)
        parsed = urllib.parse.urlparse(url)
        indicator = parsed.path.split("/indicator/")[1]
        query = urllib.parse.parse_qs(parsed.query)
        page = int(query["page"][0])
        start, end = (int(part) for part in query["date"][0].split(":"))
        records = [
            {"countryiso3code": iso3, "date": str(year), "value": value}
            for (iso3, year), value in sorted(self.cells[indicator].items())
            if start <= year <= end
        ]
        records.insert(0, {"countryiso3code": "WLD", "date": str(start), "value": 1.0})
        records.append({"countryiso3code": "", "date": str(start), "value": 2.0})
        size = self.page_size or max(len(records), 1)
        pages = max(1, math.ceil(len(records) / size))
        return [{"page": page, "pages": pages, "per_page": size, "total": len(records)}, records[(page - 1) * size : page * size]]


def _api_cells(world, changes=None):
    """Copy of the world with the given ``(indicator, iso3, year): value`` cells replaced."""
    cells = {indicator: dict(values) for indicator, values in world.items()}
    for (indicator, iso3, year), value in (changes or {}).items():
        cells[indicator][(iso3, year)] = value
    return cells


def _run(tmp_path: Path, stored, api_cells, page_size=None):
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, stored)
    fake = FakeAPI(api_cells, page_size=page_size)
    stream = io.StringIO()
    code = verify.run(raw_dir, tmp_path / "out", fetch_json=fake, stream=stream)
    differences = pd.read_csv(tmp_path / "out" / verify.DIFFERENCES_FILE, keep_default_na=False)
    return code, differences, stream.getvalue(), fake


def test_constants_are_those_of_the_panel_module():
    assert verify.GROUPS == panel.GROUPS
    assert verify.INDICATORS == tuple(panel.INDICATORS)
    for group in verify.GROUPS:
        for indicator in verify.INDICATORS:
            assert verify.stored_file_name(group, indicator) == panel.wdi_filename(group, indicator)


def test_identical_data_agrees_and_writes_only_the_header(tmp_path):
    world = _world()
    code, differences, text, _ = _run(tmp_path, world, _api_cells(world))
    assert code == 0
    assert differences.empty and list(differences.columns) == list(verify.COLUMNS)
    assert "every compared cell agrees" in text
    assert "total" in text


def test_compared_cells_are_counted_per_indicator(tmp_path):
    world = _world()
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, world)
    stored = pd.concat([verify.load_stored(raw_dir, g, "ST.INT.ARVL") for g in verify.GROUPS], ignore_index=True)
    api = verify.fetch_series("https://example.org", "ST.INT.ARVL", 2000, 2002, FakeAPI(_api_cells(world)))
    detail, counts = verify.compare_series("ST.INT.ARVL", stored, api, ECONOMIES, 1e-9)
    assert counts == {"compared": 9, "equal": 9, "different": 0, "only_file": 0, "only_api": 0, "empty_both": 0}
    assert detail.empty


def test_a_changed_cell_is_reported_and_fails(tmp_path):
    world = _world()
    api = _api_cells(world, {("NY.GDP.MKTP.CD", "BBB", 2001): world["NY.GDP.MKTP.CD"][("BBB", 2001)] * (1 + 1e-6)})
    code, differences, _, _ = _run(tmp_path, world, api)
    assert code == 1
    assert len(differences) == 1
    row = differences.iloc[0]
    assert (row["indicator"], row["group"], row["iso3"], int(row["year"]), row["status"]) == (
        "NY.GDP.MKTP.CD",
        "G2",
        "BBB",
        2001,
        "different",
    )
    assert math.isclose(float(row["relative_difference"]), 1e-6, rel_tol=1e-3)


@pytest.mark.parametrize("factor, expected_code", [(1 + 1e-12, 0), (1 + 5e-10, 0), (1 + 2e-9, 1), (1 - 2e-9, 1)])
def test_the_relative_tolerance_is_applied_to_the_larger_value(tmp_path, factor, expected_code):
    world = _world()
    api = _api_cells(world, {("SP.POP.TOTL", "CCC", 2000): world["SP.POP.TOTL"][("CCC", 2000)] * factor})
    code, _, _, _ = _run(tmp_path, world, api)
    assert code == expected_code


def test_a_value_only_in_the_file_is_reported(tmp_path):
    world = _world()
    api = _api_cells(world, {("IS.AIR.PSGR", "AAA", 2002): None})
    code, differences, _, _ = _run(tmp_path, world, api)
    assert code == 1
    assert list(differences["status"]) == ["only in file"]
    assert differences.iloc[0]["iso3"] == "AAA" and int(differences.iloc[0]["year"]) == 2002


def test_a_value_only_in_the_api_is_reported(tmp_path):
    world = _world()
    stored = _api_cells(world, {("ST.INT.RCPT.CD", "CCC", 2001): None})
    code, differences, _, _ = _run(tmp_path, stored, _api_cells(world))
    assert code == 1
    assert list(differences["status"]) == ["only in API"]
    assert differences.iloc[0]["indicator"] == "ST.INT.RCPT.CD"


def test_a_cell_empty_on_both_sides_is_counted_and_passes(tmp_path):
    world = _world()
    stored = _api_cells(world, {("SP.POP.TOTL", "BBB", 2000): None})
    api = _api_cells(world, {("SP.POP.TOTL", "BBB", 2000): None})
    code, differences, text, _ = _run(tmp_path, stored, api)
    assert code == 0 and differences.empty
    assert "total" in text


def test_economies_outside_the_metadata_are_not_compared_from_the_api(tmp_path):
    world = _world()
    api = _api_cells(world)
    api["ST.INT.ARVL"][("ZZZ", 2000)] = 5.0
    code, differences, _, _ = _run(tmp_path, world, api)
    assert code == 0 and differences.empty


def test_pages_of_the_api_are_all_read(tmp_path):
    world = _world()
    code, differences, _, fake = _run(tmp_path, world, _api_cells(world), page_size=2)
    assert code == 0 and differences.empty
    assert any("page=2" in url for url in fake.urls)


def test_requests_name_the_indicator_format_and_the_stored_years(tmp_path):
    world = _world()
    _, _, _, fake = _run(tmp_path, world, _api_cells(world))
    url = verify.api_url("https://api.worldbank.org/v2/", "ST.INT.RCPT.CD", 2000, 2002, 1)
    assert url == "https://api.worldbank.org/v2/country/all/indicator/ST.INT.RCPT.CD?format=json&per_page=20000&page=1&date=2000:2002"
    assert len(fake.urls) == len(verify.INDICATORS)
    assert all("date=2000:2002" in u and "format=json" in u for u in fake.urls)


def test_an_unexpected_answer_of_the_api_is_an_error():
    def fake(url):
        return {"message": [{"id": "120", "key": "Invalid value"}]}

    with pytest.raises(RuntimeError, match="unexpected answer"):
        verify.fetch_series("https://example.org", "ST.INT.ARVL", 2000, 2002, fake)


def test_the_api_must_not_repeat_a_cell():
    def fake(url):
        return [{"page": 1, "pages": 1}, [{"countryiso3code": "AAA", "date": "2000", "value": 1}, {"countryiso3code": "AAA", "date": "2000", "value": 2}]]

    with pytest.raises(RuntimeError, match="repeats"):
        verify.fetch_series("https://example.org", "ST.INT.ARVL", 2000, 2002, fake)


def test_a_stored_value_that_is_not_a_number_is_an_error(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "wdi_G1_ST.INT.ARVL.csv").write_text("iso3,year,value\nAAA,2000,lots\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a number"):
        verify.load_stored(raw_dir, "G1", "ST.INT.ARVL")


def test_repeated_economy_codes_in_the_metadata_are_an_error(tmp_path):
    path = tmp_path / "country_metadata.csv"
    path.write_text("iso3,name\nAAA,A\nAAA,A again\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repeated"):
        verify.load_economies(path)


def test_a_missing_stored_file_stops_the_check_before_the_download(tmp_path):
    world = _world()
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, world)
    (raw_dir / verify.stored_file_name("G3", "SP.POP.TOTL")).unlink()

    def forbidden(url):
        raise AssertionError("no download expected")

    with pytest.raises(FileNotFoundError):
        verify.run(raw_dir, tmp_path / "out", fetch_json=forbidden, stream=io.StringIO())


def test_main_reports_an_incomplete_check_with_status_two(tmp_path, monkeypatch, capsys):
    world = _world()
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, world)

    def unavailable(url):
        raise RuntimeError("download failed after three attempts: no route")

    monkeypatch.setattr(verify, "download_json", unavailable)
    code = verify.main(["--out", str(tmp_path / "out"), "--raw-dir", str(raw_dir)])
    assert code == 2
    assert "Check not completed" in capsys.readouterr().err


def test_main_needs_an_output_directory():
    with pytest.raises(SystemExit):
        verify.main([])


def test_the_output_directory_is_created(tmp_path):
    world = _world()
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, world)
    target = tmp_path / "nested" / "checks"
    verify.run(raw_dir, target, fetch_json=FakeAPI(_api_cells(world)), stream=io.StringIO())
    assert (target / verify.DIFFERENCES_FILE).is_file()
