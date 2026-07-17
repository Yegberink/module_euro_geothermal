"""Focused tests for the study-wide potential tables."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import overall_potential  # noqa: E402


def test_lowest_lcoe_potential_selects_cellwise_winner():
    """The portfolio must use potential from each cell's cheapest technology."""
    lcoe = {
        "orc": np.array([[50.0, 80.0], [np.nan, 60.0]]),
        "flash": np.array([[70.0, 40.0], [30.0, np.nan]]),
    }
    potential = {
        "orc": np.array([[1e6, 2e6], [np.nan, 4e6]]),
        "flash": np.array([[5e6, 6e6], [7e6, np.nan]]),
    }
    assert overall_potential.lowest_lcoe_potential_tw(lcoe, potential) == 18.0


def test_potential_summary_reports_baseline_and_range():
    """The compact table must expose its central value and sensitivity envelope."""
    by_scenario = pd.DataFrame(
        {
            "scenario": ["base", "low", "high"],
            "technology": ["orc", "orc", "orc"],
            "potential_tw": [6.0, 2.0, 9.0],
        }
    )
    summary = overall_potential.potential_summary_table(by_scenario, "base").iloc[0]
    assert summary["baseline_potential_tw"] == 6.0
    assert summary["sensitivity_min_potential_tw"] == 2.0
    assert summary["sensitivity_max_potential_tw"] == 9.0
    assert summary["sensitivity_min_scenario"] == "low"
    assert summary["sensitivity_max_scenario"] == "high"
    assert summary["n_sensitivity_scenarios"] == 2
