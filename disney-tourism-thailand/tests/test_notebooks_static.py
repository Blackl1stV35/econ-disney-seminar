"""Static checks of the shipped notebooks and of the repository files (nothing is executed)."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
ANALYSIS_NOTEBOOKS = sorted(p for p in NOTEBOOK_DIR.glob("0*.ipynb") if not p.name.startswith("01_"))
SECRET_PATTERN = re.compile(r"github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".ipynb", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".gitignore"}
#: Directories that hold a virtual environment, a tool cache or executed copies, not files of the repository.
SKIPPED_DIRECTORIES = {".venv", "venv", "env", "site-packages", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "executed"}
#: Directory of the data files. The text scan does not open the files below it: they hold the data of the
#: analysis, not text of the repository.
DATA_DIRECTORY = "data"
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
    assert "DTT_RUN_REAL=1" in first_markdown.source
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


def _skipped_directory(directory: Path) -> bool:
    """True for a directory that holds a virtual environment, a tool cache or executed copies."""
    return directory.name in SKIPPED_DIRECTORIES or (directory / "pyvenv.cfg").is_file()


def repository_files(root: Path) -> list[Path]:
    """Return the files below ``root`` as paths relative to it.

    Directories are skipped by their name below ``root`` and by a ``pyvenv.cfg``
    file inside them, so the names of the directories above ``root`` play no part.
    """
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        here = Path(current)
        dirnames[:] = sorted(name for name in dirnames if not _skipped_directory(here / name))
        found.extend((here / name).relative_to(root) for name in filenames)
    return sorted(found)


def _data_file(relative: Path) -> bool:
    """True for a file below the data directory, which the text scan does not open."""
    return relative.parts[0] == DATA_DIRECTORY


def offending_text_files(root: Path) -> list[str]:
    """List the text files below ``root`` that hold a token-like string or an em dash.

    The files below the data directory are skipped without being opened.
    """
    offenders = []
    for relative in repository_files(root):
        if _data_file(relative):
            continue
        path = root / relative
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_PATTERN.search(text):
            offenders.append(f"{relative.as_posix()}: token-like string")
        if EM_DASH in text and path.suffix != ".csv":
            offenders.append(f"{relative.as_posix()}: em dash")
    return offenders


def test_repository_text_files_have_no_em_dash_and_no_secrets():
    assert offending_text_files(ROOT) == []


def test_the_text_scan_does_not_open_the_data_files(tmp_path, monkeypatch):
    repo = tmp_path / "copy"
    for relative in ("data/reference/values.json", "data/raw/wdi/fetch_log_G1.json", "data/processed/cases_catalogue.csv"):
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text(f'{{"note": "{EM_DASH}"}}', encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text(f"A note {EM_DASH}.\n", encoding="utf-8")
    opened: list[str] = []
    read_text = Path.read_text

    def recording(self, *args, **kwargs):
        opened.append(self.name)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    assert offending_text_files(repo) == ["docs/note.md: em dash"]
    assert not {"values.json", "fetch_log_G1.json", "cases_catalogue.csv"} & set(opened)
    assert "note.md" in opened


def test_repository_scan_ignores_virtual_environments_and_finds_the_repository_files(tmp_path):
    repo = tmp_path / "env" / "copy"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "note.md").write_text(f"A note {EM_DASH} with an em dash.\n", encoding="utf-8")
    venv = repo / ".venv"
    (venv / "lib" / "site-packages" / "pkg").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / "lib" / "site-packages" / "pkg" / "module.md").write_text(f"Third-party {EM_DASH} text.\n", encoding="utf-8")
    vendored = repo / "tools" / "other_python"
    vendored.mkdir(parents=True)
    (vendored / "pyvenv.cfg").write_text("", encoding="utf-8")
    (vendored / "readme.md").write_text(f"{EM_DASH}\n", encoding="utf-8")
    (repo / "venv_copy" / "env").mkdir(parents=True)
    (repo / "venv_copy" / "env" / "notes.md").write_text(f"{EM_DASH}\n", encoding="utf-8")

    assert offending_text_files(repo) == ["docs/note.md: em dash"]
    listed = [path.as_posix() for path in repository_files(repo)]
    assert "docs/note.md" in listed
    assert not any(part in SKIPPED_DIRECTORIES for path in repository_files(repo) for part in path.parts)
    assert not any(path.parts[0] == ".venv" or "pyvenv.cfg" in path.parts for path in repository_files(repo))
