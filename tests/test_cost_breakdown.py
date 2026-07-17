"""Focused tests for the geographic LCOE cost breakdown."""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

SCRIPTS = Path(__file__).parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import cost_breakdown  # noqa: E402


def _technology_shares(technology, offset):
    values = np.array([[10.0, 20.0], [30.0, np.nan]]) + offset
    return xr.Dataset(
        {
            component: (("y", "x"), values)
            for component, _ in cost_breakdown.COMPONENTS
        }
    ).expand_dims(technology=[technology])


def test_technology_summaries_remain_separate():
    """Each technology should retain its own mean and geographic range."""
    summaries = cost_breakdown.summarize_technology_shares(
        [_technology_shares("orc", 0.0), _technology_shares("flash", 10.0)]
    )
    assert {
        key: summaries["orc"]["drilling"][key] for key in ("mean", "min", "max")
    } == {"mean": 20.0, "min": 10.0, "max": 30.0}
    assert {
        key: summaries["flash"]["drilling"][key] for key in ("mean", "min", "max")
    } == {"mean": 30.0, "min": 20.0, "max": 40.0}


def test_cost_breakdown_figure(tmp_path):
    """Technology-specific component ranges should be plottable."""
    shares = [_technology_shares("orc", 0.0), _technology_shares("flash", 10.0)]
    output = tmp_path / "cost_breakdown.png"
    cost_breakdown.make_figure(shares, output)
    assert output.stat().st_size > 0
