"""End-to-end run of the analysis notebooks on a simulated world.

Notebooks 00, 02, 03, 04 and 05 are executed in this order with ``DTT_WORLD=sim`` in directories outside the
repository.  The tests then check that every file of the file contract exists with its columns and that the run left
the files of the repository unchanged.  All data are simulated.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
STAGES = [
    "00_build_global_panel",
    "02_case_effects",
    "03_feature_importance",
    "04_optimal_transport",
    "05_thailand_impact",
]
CELL_TIMEOUT_SECONDS = 1200
SKIPPED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}

RESULT_FILES = {
    "00_build_global_panel": ["panel_coverage.csv", "donor_eligibility.csv"],
    "02_case_effects": [
        "episodes.csv",
        "feasibility.csv",
        "case_effects.csv",
        "case_placebos.csv",
        "case_effects_robustness.csv",
        "reference_features.csv",
        "case_features.csv",
        "case_features_imputed.csv",
        "imputation_report.csv",
    ],
    "03_feature_importance": ["feature_importance.csv", "importance_diagnostics.json", "transport_weights.csv"],
    "04_optimal_transport": [
        "loco_table.csv",
        "loco_summary.csv",
        "loco_ratios.csv",
        "transport_plan.csv",
        "thailand_transport.json",
        "thailand_transport_draws.csv",
        "thailand_null_draws.csv",
        "overlap_test.json",
        "loco_fold_weights.csv",
    ],
    "05_thailand_impact": ["route_decision.json", "thailand_impact.csv", "thailand_summary.json"],
}

COLUMNS = {
    "case_effects.csv": [
        "case_id", "iso3", "opening_year", "window_end", "att_pp", "att_rel_pct", "placebo_sd_pp", "placebo_sd_rel_pct",
        "p_gap_rank", "fit_exact", "att_pp_min", "att_pp_max", "pre_fit_ok", "sample_primary", "first_episode",
        "parks_only", "category", "capex_pct_gdp", "n_openings",
    ],
    "episodes.csv": ["case_id", "iso3", "opening_year", "n_openings", "member_case_ids", "member_investment_usd_bn"],
    "feature_importance.csv": [
        "outcome", "sample", "feature", "permutation", "permutation_normalised", "coefficient_path",
        "selection_frequency", "boot_median", "boot_p10", "boot_p90", "cluster", "cluster_importance",
    ],
    "transport_weights.csv": ["feature", "weight_importance", "weight_shrunk", "weight_uniform"],
    "thailand_transport_draws.csv": ["draw", "predictive_pp", "predictive_rel_pct", "transport_pp", "transport_rel_pct"],
    "thailand_null_draws.csv": ["null_pp", "null_rel_pct"],
    "panel_coverage.csv": ["n_holes", "n_negative"],
    "donor_eligibility.csv": ["iso3", "n_years_outcome_1995_2019", "complete_1995_2019"],
    "loco_fold_weights.csv": ["held_out_group", "feature", "weight"],
}


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Return the size and modification time of every file below ``root`` outside the cache directories."""
    state = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIPPED_PARTS]
        for name in files:
            path = Path(folder) / name
            stat = path.stat()
            state[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return state


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Execute the notebooks in order on a simulated world and return the directories used."""
    base = tmp_path_factory.mktemp("pipeline")
    kernels = base / "jupyter" / "kernels" / "dtt-test"
    kernels.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "dtt-test",
        "language": "python",
    }
    (kernels / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")

    directories = {"world": base / "world", "results": base / "results", "figures": base / "figures"}
    environment = {
        "JUPYTER_PATH": str(base / "jupyter"),
        "DTT_WORLD": "sim",
        "DTT_SIM_DIR": str(directories["world"]),
        "DTT_RESULTS_DIR": str(directories["results"]),
        "DTT_FIGURE_DIR": str(directories["figures"]),
        "DTT_GUARD": "0",
    }
    saved = {key: os.environ.get(key) for key in [*environment, "DTT_RUN_REAL"]}
    os.environ.update(environment)
    os.environ.pop("DTT_RUN_REAL", None)
    before = _snapshot(ROOT)
    try:
        for stage in STAGES:
            notebook = nbformat.read(NOTEBOOK_DIR / f"{stage}.ipynb", as_version=4)
            client = nbclient.NotebookClient(
                notebook,
                timeout=CELL_TIMEOUT_SECONDS,
                kernel_name="dtt-test",
                resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
            )
            client.execute()
        after = _snapshot(ROOT)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {**directories, "before": before, "after": after}


def test_the_run_leaves_the_repository_unchanged(pipeline):
    assert pipeline["after"] == pipeline["before"]


@pytest.mark.parametrize("stage", STAGES)
def test_stage_writes_the_files_of_the_contract(pipeline, stage):
    for name in RESULT_FILES[stage]:
        path = pipeline["results"] / name
        assert path.is_file() and path.stat().st_size > 0, f"{stage} did not write {name}"
    if stage == "00_build_global_panel":
        assert (pipeline["world"] / "processed" / "global_panel.csv").is_file()


@pytest.mark.parametrize("name, columns", sorted(COLUMNS.items()))
def test_tables_hold_the_contract_columns(pipeline, name, columns):
    table = pd.read_csv(pipeline["results"] / name)
    missing = [c for c in columns if c not in table.columns]
    assert not missing, f"{name} lacks {missing}"
    assert len(table) > 0


def test_figures_are_listed_in_the_inventories_and_exist(pipeline):
    inventories = sorted(pipeline["results"].glob("figure_inventory_*.csv"))
    assert {p.stem.removeprefix("figure_inventory_") for p in inventories} == set(STAGES)
    for path in inventories:
        inventory = pd.read_csv(path)
        assert len(inventory) > 0
        for name in inventory["file"]:
            figure = pipeline["figures"] / name
            assert figure.is_file() and figure.stat().st_size > 1000, f"{name} is missing or empty"
        assert inventory["x labels"].notna().all() and inventory["y labels"].notna().all()


def test_effect_table_is_consistent(pipeline):
    effects = pd.read_csv(pipeline["results"] / "case_effects.csv")
    assert effects["case_id"].is_unique
    assert effects["sample_primary"].sum() >= 1
    assert (effects.loc[effects["sample_primary"], "pre_fit_ok"]).all()
    assert not effects.loc[effects["fit_exact"], "pre_fit_ok"].any()
    inside = (effects["att_pp"] >= effects["att_pp_min"] - 1e-9) & (effects["att_pp"] <= effects["att_pp_max"] + 1e-9)
    assert inside.all()
    assert (effects["placebo_sd_pp"] > 0).all()
    assert effects["p_gap_rank"].between(0.0, 1.0).all()


def test_route_decision_is_stated_before_the_numbers(pipeline):
    decision = json.loads((pipeline["results"] / "route_decision.json").read_text(encoding="utf-8"))
    assert {"primary", "passed", "details"} <= set(decision)
    assert decision["primary"] in {"ot_importance", "ambient"}
    assert isinstance(decision["passed"], dict) and len(decision["passed"]) == 3
    assert all(isinstance(value, bool) for value in decision["passed"].values())
    summary = json.loads((pipeline["results"] / "thailand_summary.json").read_text(encoding="utf-8"))
    assert summary
    impact = pd.read_csv(pipeline["results"] / "thailand_impact.csv")
    assert len(impact) > 0
    assert not np.isinf(impact.select_dtypes("number").to_numpy(dtype=float)).any()


def test_predictive_draws_are_finite_and_ordered(pipeline):
    draws = pd.read_csv(pipeline["results"] / "thailand_transport_draws.csv")
    assert draws[["predictive_pp", "predictive_rel_pct"]].notna().all().all()
    lo, hi = draws["predictive_pp"].quantile([0.05, 0.95])
    assert lo < hi
    null = pd.read_csv(pipeline["results"] / "thailand_null_draws.csv")
    assert null["null_pp"].notna().all() and null["null_pp"].std() > 0
