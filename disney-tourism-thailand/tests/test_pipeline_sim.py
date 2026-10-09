"""End-to-end run of the analysis notebooks on a simulated world.

Notebooks 00, 02, 03, 04 and 05 are executed in this order with ``DTT_WORLD=sim`` in directories outside the
repository.  The tests then check that every file of the file contract exists with its columns and that the run left
the files of the repository unchanged.  All data are simulated.  A second fixture runs notebook 05 again on a copy of
the results with the importance verdict, the case minimum and the support result forced to hold, so that the
importance route of the decision is executed although a simulated world with a handful of episodes never reaches it.
"""

from __future__ import annotations

import json
import os
import shutil
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
        "case_features_first_opening_imputed.csv",
        "imputation_report.csv",
        "imputation_report_first_opening.csv",
    ],
    "03_feature_importance": [
        "feature_importance.csv",
        "importance_diagnostics.json",
        "transport_weights.csv",
        "verdict_stability.csv",
    ],
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
        "thailand_target_baseline.csv",
        "thailand_scenario_draws.csv",
        "transport_sensitivity.csv",
        "transport_usage.csv",
    ],
    "05_thailand_impact": ["route_decision.json", "ambient_loeo.csv", "thailand_impact.csv", "thailand_summary.json"],
}
ROUTE_CRITERIA = {"importance_usable", "loco_rmse", "enough_cases", "target_supported"}

COLUMNS = {
    "case_effects.csv": [
        "case_id", "iso3", "opening_year", "window_end", "att_pp", "att_rel_pct", "placebo_sd_pp", "placebo_sd_rel_pct",
        "p_gap_rank", "fit_exact", "att_pp_min", "att_pp_max", "att_pp_lp_min", "att_pp_lp_max", "n_placebos",
        "n_placebos_comparable", "n_placebos_exact", "fit_tolerance", "pre_fit_ok", "sample_primary", "first_episode",
        "parks_only", "category", "capex_pct_gdp", "capex_first_pct_gdp", "n_openings",
    ],
    "case_features_first_opening_imputed.csv": ["case_id", "iso3", "opening_year", "capex_first_pct_gdp"],
    "verdict_stability.csv": [
        "outcome", "sample", "seed", "cv_r2_stack", "perm_p_value", "split_half_correlation", "n_cases", "usable",
    ],
    "thailand_impact.csv": [
        "scenario", "route", "baseline_year", "scaling", "q5", "q50", "q95", "primary", "interval_lower",
        "interval_upper", "interval_kind", "interval_level", "central", "central_kind", "importance_route_supported",
    ],
    "thailand_target_baseline.csv": ["year", "gdp_usd_bn", "receipts_usd_bn", "receipts_pct_gdp", "receipts_basis"],
    "thailand_scenario_draws.csv": ["scenario", "draw", "predictive_pp", "predictive_rel_pct"],
    "episodes.csv": ["case_id", "iso3", "opening_year", "n_openings", "member_case_ids", "member_investment_usd_bn"],
    "feature_importance.csv": [
        "outcome", "sample", "feature", "permutation", "permutation_normalised", "coefficient_path",
        "selection_frequency", "boot_median", "boot_p10", "boot_p90", "cluster", "cluster_importance",
        "n_resamples_informative", "n_resamples",
    ],
    "transport_weights.csv": ["feature", "weight_importance", "weight_shrunk", "weight_uniform"],
    "thailand_transport_draws.csv": ["draw", "predictive_pp", "predictive_rel_pct", "transport_pp", "transport_rel_pct"],
    "thailand_null_draws.csv": ["null_pp", "null_rel_pct"],
    "panel_coverage.csv": ["n_holes", "n_negative"],
    "donor_eligibility.csv": ["iso3", "n_years_outcome_1995_2019", "complete_1995_2019"],
    "loco_fold_weights.csv": ["held_out_group", "feature", "weight", "fitted"],
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
    return {**directories, "before": before, "after": after, "environment": environment}


FORCING_CELL = """import types

from dtt import impact as _impact
from dtt import workflow as _workflow

_impact.DEFAULT_ROUTE_RULES = types.MappingProxyType({**_impact.DEFAULT_ROUTE_RULES, "min_cases": 5})
_read_json = _workflow.RunContext.read_json


def _forced_read_json(self, name):
    record = _read_json(self, name)
    if name == "importance_diagnostics":
        record["att_pp|primary"]["usable"] = True
    if name == "thailand_transport":
        record["support"]["supported"] = True
    return record


_workflow.RunContext.read_json = _forced_read_json
"""


@pytest.fixture(scope="module")
def forced_pipeline(pipeline, tmp_path_factory):
    """Run notebook 05 on a copy of the results with the verdict, the case minimum and the support forced to hold."""
    base = tmp_path_factory.mktemp("forced")
    results = base / "results"
    figures = base / "figures"
    shutil.copytree(pipeline["results"], results)
    shutil.copytree(pipeline["figures"], figures)
    environment = {**pipeline["environment"], "DTT_RESULTS_DIR": str(results), "DTT_FIGURE_DIR": str(figures)}
    saved = {key: os.environ.get(key) for key in environment}
    notebook = nbformat.read(NOTEBOOK_DIR / "05_thailand_impact.ipynb", as_version=4)
    first_code = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code")
    notebook.cells.insert(first_code + 1,nbformat.v4.new_code_cell(FORCING_CELL))
    os.environ.update(environment)
    try:
        client = nbclient.NotebookClient(
            notebook,
            timeout=CELL_TIMEOUT_SECONDS,
            kernel_name="dtt-test",
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        )
        client.execute()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"results": results, "figures": figures}


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
    assert isinstance(decision["passed"], dict) and set(decision["passed"]) == ROUTE_CRITERIA
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


def test_route_rule_states_four_criteria_and_a_minimum_of_twenty_cases(pipeline):
    decision = json.loads((pipeline["results"] / "route_decision.json").read_text(encoding="utf-8"))
    assert decision["rule"]["min_cases"] == 20
    assert decision["rule"]["importance_usable_required"] and decision["rule"]["target_supported_required"]
    assert len(decision["rule"]["sha256"]) == 64
    assert decision["primary"] == ("ot_importance" if all(decision["passed"].values()) else "ambient")
    assert decision["passed"]["enough_cases"] == (decision["inputs"]["n_validated_episodes"] >= 20)


def test_importance_diagnostics_report_resamples_and_verdict_stability(pipeline):
    diagnostics = json.loads((pipeline["results"] / "importance_diagnostics.json").read_text(encoding="utf-8"))
    entry = diagnostics["att_pp|primary"]
    assert {"usable", "reasons", "n_resamples", "n_resamples_informative", "no_signal", "verdict_stability"} <= set(entry)
    assert 0 <= entry["n_resamples_informative"] <= entry["n_resamples"]
    stability = entry["verdict_stability"]
    assert {"by_seed", "n_seeds", "n_usable", "share_usable"} <= set(stability)
    assert len(stability["by_seed"]) == stability["n_seeds"]
    table = pd.read_csv(pipeline["results"] / "verdict_stability.csv")
    selected = (table["outcome"] == "att_pp") & (table["sample"] == "primary")
    assert int(table.loc[selected, "usable"].sum()) == stability["n_usable"]
    importance = pd.read_csv(pipeline["results"] / "feature_importance.csv")
    assert (importance["n_resamples_informative"] <= importance["n_resamples"]).all()


def test_first_opening_cost_is_defined_for_every_episode(pipeline):
    effects = pd.read_csv(pipeline["results"] / "case_effects.csv")
    first = pd.read_csv(pipeline["results"] / "case_features_first_opening_imputed.csv")
    assert set(first["case_id"]) == set(effects["case_id"])
    assert first["capex_first_pct_gdp"].notna().all()
    primary = effects[effects["sample_primary"]]
    assert primary.groupby("iso3")["first_episode"].sum().le(1).all()


def test_limits_are_unwidened_and_inside_the_widened_range(pipeline):
    effects = pd.read_csv(pipeline["results"] / "case_effects.csv")
    solved = effects[effects["att_pp_lp_min"].notna()]
    assert solved["fit_exact"].all()
    assert (solved["att_pp_lp_min"] <= solved["att_pp_lp_max"] + 1e-9).all()
    assert (solved["att_pp_lp_min"] >= solved["att_pp_min"] - 1e-9).all()
    assert (solved["att_pp_lp_max"] <= solved["att_pp_max"] + 1e-9).all()
    assert (effects["n_placebos_exact"] <= effects["n_placebos_comparable"]).all()
    assert (effects["n_placebos_comparable"] <= effects["n_placebos"]).all()


def test_the_target_description_of_notebook_04_holds_the_support_check(pipeline):
    record = json.loads((pipeline["results"] / "thailand_transport.json").read_text(encoding="utf-8"))
    assert {"support", "scenarios", "input_digests", "n_nonconverged_bootstrap"} <= set(record)
    assert isinstance(record["support"]["supported"], bool)
    assert record["n_nonconverged_bootstrap"] == 0
    assert {"case_effects", "case_features_imputed"} <= set(record["input_digests"])
    assert {"n_log_domain", "n_log_domain_rel", "n_nonconverged"} <= set(record["validation"])
    assert {"n_log_domain_bootstrap", "plan_log_domain"} <= set(record)
    assert record["n_log_domain_bootstrap"] >= 0 and isinstance(record["plan_log_domain"], bool)
    sensitivity = pd.read_csv(pipeline["results"] / "transport_sensitivity.csv")
    assert "log_domain" in sensitivity.columns


def test_file_stamps_cover_the_saved_results_without_paths(pipeline):
    text = (pipeline["results"] / "file_stamps.json").read_text(encoding="utf-8")
    stamps = json.loads(text)
    assert stamps["files"]
    assert str(pipeline["results"]) not in text and str(pipeline["world"]) not in text


def test_forced_verdict_runs_the_importance_route(forced_pipeline):
    decision = json.loads((forced_pipeline["results"] / "route_decision.json").read_text(encoding="utf-8"))
    assert decision["primary"] == "ot_importance"
    assert set(decision["passed"]) == ROUTE_CRITERIA and all(decision["passed"].values())
    impact = pd.read_csv(forced_pipeline["results"] / "thailand_impact.csv")
    primary = impact[impact["primary"].astype(bool)]
    assert len(primary) > 0 and set(primary["route"]) == {"ot_importance"}
    assert (primary["interval_lower"] < primary["interval_upper"]).all()
    summary = json.loads((forced_pipeline["results"] / "thailand_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_route"] == "ot_importance"


def _changed_input_run(pipeline, tmp_path, stage, change):
    """Execute one notebook on copies of the world and the results after ``change(world_directory)`` altered an input."""
    world, results, figures = tmp_path / "world", tmp_path / "results", tmp_path / "figures"
    shutil.copytree(pipeline["world"], world)
    shutil.copytree(pipeline["results"], results)
    shutil.copytree(pipeline["figures"], figures)
    change(world)
    environment = {
        **pipeline["environment"],
        "DTT_SIM_DIR": str(world),
        "DTT_RESULTS_DIR": str(results),
        "DTT_FIGURE_DIR": str(figures),
    }
    saved = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        notebook = nbformat.read(NOTEBOOK_DIR / f"{stage}.ipynb", as_version=4)
        client = nbclient.NotebookClient(
            notebook,
            timeout=CELL_TIMEOUT_SECONDS,
            kernel_name="dtt-test",
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        )
        client.execute()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _scale_panel_receipts(world):
    path = world / "processed" / "global_panel.csv"
    panel = pd.read_csv(path)
    rows = panel["iso3"] == sorted(panel["iso3"].unique())[0]
    panel.loc[rows, "receipts_usd"] = panel.loc[rows, "receipts_usd"] * 1.3
    panel.to_csv(path, index=False, lineterminator="\n")


def _touch_baseline(world):
    with open(world / "target_baseline.csv", "a", encoding="utf-8", newline="") as handle:
        handle.write("\n")


def _touch_proposal(world):
    path = world / "target_proposal.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "stage, change",
    [
        ("02_case_effects", _scale_panel_receipts),
        ("04_optimal_transport", _scale_panel_receipts),
        ("05_thailand_impact", _touch_baseline),
        ("05_thailand_impact", _touch_proposal),
        ("05_thailand_impact", _scale_panel_receipts),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_a_changed_panel_baseline_or_proposal_stops_the_notebooks_that_read_stale_results(pipeline, tmp_path, stage, change):
    with pytest.raises(nbclient.exceptions.CellExecutionError, match="StaleResultError"):
        _changed_input_run(pipeline, tmp_path, stage, change)
