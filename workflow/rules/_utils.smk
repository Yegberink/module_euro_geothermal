"""Snakemake helper functions, configuration setup, and shared constants."""

import re

import yaml

from snakemake.io import report as report_output
from snakemake.utils import min_version, validate


min_version("9.19")


def _get_depth_category_labels():
    """Return compact labels for the configured depth categories."""
    return [
        f"{int(start/1000)}-{int(stop/1000)}km"
        for start, stop in config.get("depth_categories") or []
    ]


# Load internal resource metadata separately from model configuration.
with open(workflow.source_path("../internal/settings.yaml"), "r") as f:
    internal = yaml.safe_load(f) or {}


# Require explicit model settings for the selected calculation modes.
validate(config, workflow.source_path("../internal/config.schema.yaml"))
# Check relationships between settings after validating their types and ranges.
if config["carrier_out"] != "power":
    if config["heat_reinjection_temperature"] >= config["heat_production_temperature"]:
        raise ValueError(
            "Heat reinjection temperature must be below production temperature."
        )

curve = config["drilling_cost_curve"]
if curve not in config["drilling_cost_curve_definitions"]:
    raise ValueError(f"drilling_cost_curve {curve!r} has no matching definition.")

if config["carrier_out"] != "heat":
    previous_stop = 0.0
    for category in config.get("depth_categories", []):
        start, stop = map(float, category)
        if start >= stop:
            raise ValueError(
                f"Invalid depth category {category!r}; expected start < stop."
            )
        if start < previous_stop:
            raise ValueError("depth_categories must be ordered and non-overlapping.")
        previous_stop = stop

    previous_stop = 0.0
    for value in config.get("fixed_bins", []):
        start, stop = map(float, value.split("-", maxsplit=1))
        if start >= stop:
            raise ValueError(f"Invalid fixed LCOE bin {value!r}.")
        if start < previous_stop:
            raise ValueError("fixed_bins must be ordered and non-overlapping.")
        previous_stop = stop


# Shared rule constants.
CURL_ARGS = "--fail --silent --show-error --location --retry 5 --retry-delay 5 --retry-all-errors --continue-at -"
AUTOMATIC_DATASETS = internal["resources"]["automatic"]["datasets"]
POWER_ENABLED = config["carrier_out"] != "heat"
HEAT_ENABLED = config["carrier_out"] != "power"
CYCLES = config.get("which_cycles", []) if POWER_ENABLED else []
DEPTH_CATEGORY_LABELS = _get_depth_category_labels()
MAKE_2D_FIGURE_OUTPUTS = (
    {
        "overview_fig": report_output(
            "<results>/{shapes}/lcoe_overview.png",
            caption=workflow.source_path("../report/lcoe.rst"),
            category="Electricity potential",
            subcategory="{shapes}",
            labels={"Figure": "LCOE and conversion technology"},
        ),
        "technology_cumulative_four_panel_fig": report_output(
            "<resources>/automatic/{shapes}/figures/lcoe_potential_cumulative_four_panel.png",
            caption=workflow.source_path("../report/cumulative.rst"),
            category="Electricity potential",
            subcategory="{shapes}",
            labels={"Figure": "Cumulative potential"},
        ),
    }
    if POWER_ENABLED and config["method_2d_aggregation"] == "LCOE_minimization"
    else {}
)
CUTOFF_LABEL = (
    f"{float(config['lcoe_cutoff']):g}"
    if POWER_ENABLED
    and config["method_2d_aggregation"] == "LCOE_minimization"
    and config["method_region_aggregation"] == "cutoff"
    else None
)


ELIGIBILITY_ENABLED = config.get("eligibility", {}).get("enabled", False)
ELIGIBILITY_INPUT = (
    ["<resources>/automatic/{shapes}/eligibility/eligibility_fraction.tif"]
    if ELIGIBILITY_ENABLED
    else []
)
