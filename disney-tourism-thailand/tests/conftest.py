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
    """Simulated panel with parallel trends (SIMULATED data)."""
    from dtt import simulate

    return simulate.simulate_panel(seed=11, trend_spread=0.0)


@pytest.fixture(scope="session")
def sim_trending():
    """Simulated panel with heterogeneous donor trends (SIMULATED data)."""
    from dtt import simulate

    return simulate.simulate_panel(seed=12, trend_spread=0.5)
