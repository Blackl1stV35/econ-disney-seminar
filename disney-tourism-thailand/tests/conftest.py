"""Shared pytest configuration: import paths and fixtures."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def reference():
    """Content of data/reference/replication_reference_values.json (meta, values, tables)."""
    from dtt import replication

    return replication.load_targets()


@pytest.fixture(scope="session")
def sim_parallel():
    """Hand-built panel with the structure of the dataset and parallel trends (returns panel and truth)."""
    from handmade_panels import study_panel

    return study_panel(trend_spread=0.0, noise_seed=11)


@pytest.fixture(scope="session")
def sim_trending():
    """Hand-built panel with the structure of the dataset and heterogeneous donor trends (returns panel and truth)."""
    from handmade_panels import study_panel

    return study_panel(trend_spread=0.5, noise_seed=12)
