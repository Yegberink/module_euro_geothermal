"""Snakemake helper functions, configuration setup, and shared constants."""

import os
import re
import sys

import yaml

from snakemake.utils import min_version, validate


min_version("9.19")

sys.path.insert(0, os.path.dirname(workflow.source_path("../scripts/scenario_config.py")))
from scenario_config import build_scenario_rows, validate_module_config


def _deep_update(base, overrides):
    """Recursively apply user configuration values to the module defaults."""
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _initialize_config(defaults, overrides):
    """Fill stable defaults into the mutable Snakemake configuration."""
    merged = _deep_update(defaults, overrides)
    overrides.clear()
    overrides.update(merged)


def _get_depth_category_labels():
    """Return compact labels for the configured depth categories."""
    return [
        f"{int(start / 1000)}-{int(stop / 1000)}km"
        for start, stop in config.get("depth_categories") or []
    ]


# Load internal settings separately so users cannot modify them.
with open(workflow.source_path("../internal/settings.yaml"), "r") as f:
    internal = yaml.safe_load(f) or {}


# Fill stable module defaults before validation. This lets external workflows
# provide only the settings that matter for their requested target.
_initialize_config(internal.get("defaults", {}), config)
validate(config, workflow.source_path("../internal/config.schema.yaml"))
validate_module_config(config)


# Shared rule constants.
CURL_ARGS = "--fail --silent --show-error --location --retry 5 --retry-delay 5 --retry-all-errors --continue-at -"
AUTOMATIC_DATASETS = internal["resources"]["automatic"]["datasets"]
CYCLES = config["which_cycles"]
DEPTH_CATEGORY_LABELS = _get_depth_category_labels()
BASELINE_SCENARIO = config["baseline_scenario"]
BASELINE_SCENARIO_RE = re.escape(BASELINE_SCENARIO)
SCENARIO_ROWS = build_scenario_rows(config)
SCENARIOS = [row["scenario"] for row in SCENARIO_ROWS]
NON_BASELINE_SCENARIOS = [
    scenario for scenario in SCENARIOS if scenario != BASELINE_SCENARIO
]
MAKE_2D_FIGURE_OUTPUTS = (
    {
        "overview_fig": "<results>/{shapes}/{scenario}/lcoe_overview.png",
        "technology_cumulative_four_panel_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_potential_cumulative_four_panel.png",
    }
    if config["method_2d_aggregation"] == "LCOE_minimization"
    else {}
)
CUTOFF_LABEL = (
    f"{float(config['lcoe_cutoff']):g}"
    if config["method_2d_aggregation"] == "LCOE_minimization"
    and config["method_region_aggregation"] == "cutoff"
    else None
)
