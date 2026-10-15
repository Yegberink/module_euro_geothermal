"""Snakemake helper functions and shared constants."""

import re

from snakemake.io import report as report_output


def _get_depth_category_labels():
    """Return compact labels for the configured depth categories."""
    return [
        f"{int(start/1000)}-{int(stop/1000)}km"
        for start, stop in (
            config["depth_categories"]
            if POWER_ENABLED and config["method_2d_aggregation"] == "depth_categories"
            else []
        )
    ]


def additional_config_validation() -> None:
    """Check configuration relationships beyond the JSON schema."""
    if config["carrier_out"] != "power":
        if (
            config["heat_reinjection_temperature"]
            >= config["heat_production_temperature"]
        ):
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
                raise ValueError(
                    "depth_categories must be ordered and non-overlapping."
                )
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
CYCLES = config["which_cycles"] if POWER_ENABLED else []
DEPTH_CATEGORY_LABELS = _get_depth_category_labels()
MAKE_2D_FIGURE_OUTPUTS = (
    {
        "overview_fig": report_output(
            "<results>/{shapes}/lcoe_overview.png",
            category="Electricity potential",
            subcategory="{shapes}",
            labels={"Figure": "LCOE and conversion technology"},
        ),
        "technology_cumulative_four_panel_fig": report_output(
            "<resources>/automatic/{shapes}/figures/lcoe_potential_cumulative_four_panel.png",
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


ELIGIBILITY_ENABLED = config["eligibility"]["enabled"]
ELIGIBILITY_INPUT = (
    ["<resources>/automatic/{shapes}/eligibility/eligibility_fraction.tif"]
    if ELIGIBILITY_ENABLED
    else []
)


def drilling_curve_inputs(wildcards):
    """Track the selected CSV and resolve bundled paths relative to the module."""
    spec = config["drilling_cost_curve_definitions"][config["drilling_cost_curve"]]
    if spec["curve_type"] != "csv_fit_usd":
        return []
    path = spec["csv_path"]
    if path in {
        "workflow/internal/baseline_drilling_2017.csv",
        "workflow/internal/baseline_drilling_2025.csv",
    }:
        path = workflow.source_path("../internal/" + path.rsplit("/", 1)[1])
    return [path]
