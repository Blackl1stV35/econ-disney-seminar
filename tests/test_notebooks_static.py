"""Static checks of the shipped notebooks and of the repository files (nothing is executed)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
ANALYSIS_NOTEBOOKS = sorted(p for p in NOTEBOOK_DIR.glob("0*.ipynb") if not p.name.startswith("01_"))
SECRET_PATTERN = re.compile(r"github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".ipynb", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".gitignore"}
SKIPPED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}
EM_DASH = chr(0x2014)

BOOTSTRAP = """import sys
from pathlib import Path

_here = Path.cwd().resolve()
_root = next(p for p in (_here, *_here.parents) if (p / "src" / "dtt" / "__init__.py").is_file())
sys.path.insert(0, str(_root / "src"))

from dtt import workflow

ctx = workflow.setup_run("{stage}")
ctx.describe()"""


def _read(path: Path):
    return nbformat.read(path, as_version=4)


def test_five_analysis_notebooks_are_present():
    assert [p.stem for p in ANALYSIS_NOTEBOOKS] == [
        "00_build_global_panel",
        "02_case_effects",
        "03_feature_importance",
        "04_optimal_transport",
        "05_thailand_impact",
    ]


@pytest.mark.parametrize("path", ANALYSIS_NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_and_has_no_stored_outputs(path):
    notebook = _read(path)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None


@pytest.mark.parametrize("path", ANALYSIS_NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_opens_with_the_bootstrap_cell_and_the_description(path):
    notebook = _read(path)
    first_code = next(c for c in notebook.cells if c.cell_type == "code")
    assert first_code.source.strip() == BOOTSTRAP.format(stage=path.stem)
    first_markdown = next(c for c in notebook.cells if c.cell_type == "markdown")
    for heading in ("**Purpose.**", "**Input files.**", "**Output files.**", "**How to run.**", "**Run time.**"):
        assert heading in first_markdown.source, f"{path.stem} lacks {heading}"
    assert "DTT_RUN_REAL=1" in first_markdown.source and "DTT_WORLD=sim" in first_markdown.source
    last_markdown = [c for c in notebook.cells if c.cell_type == "markdown"][-1]
    assert last_markdown.source.lstrip().startswith("## Scope and limits")


@pytest.mark.parametrize("path", ANALYSIS_NOTEBOOKS, ids=lambda p: p.stem)
def test_code_cells_parse(path):
    for index, cell in enumerate(_read(path).cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{path.stem} cell {index}")


@pytest.mark.parametrize("path", sorted(NOTEBOOK_DIR.glob("*.ipynb")), ids=lambda p: p.stem)
def test_notebook_text_has_no_em_dash(path):
    text = "\n".join(cell.source for cell in _read(path).cells)
    assert EM_DASH not in text


def test_repository_text_files_have_no_em_dash_and_no_secrets():
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIPPED_PARTS for part in path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_PATTERN.search(text):
            offenders.append(f"{path.relative_to(ROOT)}: token-like string")
        if EM_DASH in text and path.suffix != ".csv":
            offenders.append(f"{path.relative_to(ROOT)}: em dash")
    assert not offenders, offenders
