"""Snakemake helper functions and shared constants."""

import re

from snakemake.io import report as report_output


def additional_config_validation(config=config) -> None:
    """Check configuration relationships beyond the JSON schema."""
    if config["carrier_out"] != "power":
        if (
            config["heat_reinjection_temperature"]
            >= config["heat_production_temperature"]
        ):
            raise ValueError(
                "Heat reinjection temperature must be below production temperature."
            )

    if config["carrier_out"] != "heat":
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
MAKE_2D_FIGURE_OUTPUTS = (
    {
        "overview_fig": report_output(
            "<results>/{shapes}/{scenario}/figures/lcoe_overview.pdf",
            category="power",
            subcategory="{shapes}/{scenario}",
            labels={"Figure": "LCOE and conversion technology"},
        ),
        "potential_overview_fig": report_output(
            "<results>/{shapes}/{scenario}/figures/potential_overview.pdf",
            category="power",
            subcategory="{shapes}/{scenario}",
            labels={"Figure": "Potential at minimum LCOE"},
        ),
    }
    if POWER_ENABLED
    else {}
)
ELIGIBILITY_ENABLED = config["eligibility"]["enabled"]
ELIGIBILITY_INPUT = (
    ["<resources>/automatic/{shapes}/eligibility/eligibility_fraction.tif"]
    if ELIGIBILITY_ENABLED
    else []
)


DRILLING_CSV = (
    [
        workflow.source_path(
            {
                "getem_2017": "../internal/baseline_drilling_2017.csv",
                "getem_2025": "../internal/baseline_drilling_2025.csv",
            }[config["drilling_cost_curve"]]
        )
    ]
    if config["drilling_cost_curve"] in {"getem_2017", "getem_2025"}
    else []
)


# Resolve upstream rule outputs when Snakemake builds aggregation jobs.
AGGREGATION_INPUTS = {}
AGGREGATION_OUTPUTS = {}
AGGREGATION_PARAMS = {}

if POWER_ENABLED:
    AGGREGATION_INPUTS.update(
        lcoe_tifs=lambda wc: expand(rules.make_2d.output.lcoe_tifs, **wc),
        pot_tifs=lambda wc: expand(rules.make_2d.output.pot_tifs, **wc),
        capex_tifs=lambda wc: expand(rules.make_2d.output.capex_tifs, **wc),
    )
    AGGREGATION_OUTPUTS.update(
        power=expand(
            "<resources>/automatic/{shapes}/{scenario}/documents/{cycle}_power_potential.tif",
            shapes="{shapes}",
            scenario="{scenario}",
            cycle=CYCLES,
        ),
        regional_fig=report_output(
            "<results>/{shapes}/{scenario}/figures/regional_power_potential.pdf",
            category="power",
            subcategory="{shapes}/{scenario}",
            labels={"Figure": "Regional electricity potential"},
        ),
    )
    AGGREGATION_PARAMS.update(
        method_region_aggregation=config["method_region_aggregation"],
        n_quantiles=(
            config["n_quantiles"]
            if config["method_region_aggregation"] == "quantiles"
            else None
        ),
        fixed_bins=(
            config["fixed_bins"]
            if config["method_region_aggregation"] == "fixed_bins"
            else []
        ),
        lcoe_cutoff=(
            config["lcoe_cutoff"]
            if config["method_region_aggregation"] == "cutoff"
            else None
        ),
    )

if HEAT_ENABLED:
    AGGREGATION_INPUTS.update(
        heat_capex=lambda wc: expand(rules.calculate_LCOH.output.specific_capex, **wc),
        heat_fraction=lambda wc: expand(rules.allocate_heat.output.fraction, **wc),
        heat_allocated=lambda wc: expand(rules.allocate_heat.output.allocated, **wc),
        heat_lcoh=lambda wc: expand(rules.calculate_LCOH.output.lcoh, **wc),
        heat_allocations=lambda wc: expand(
            rules.allocate_heat.output.allocations, **wc
        ),
    )
    AGGREGATION_OUTPUTS.update(
        contested_power=expand(
            "<resources>/automatic/{shapes}/{scenario}/documents/{cycle}_contested_power_potential.tif",
            shapes="{shapes}",
            scenario="{scenario}",
            cycle=CYCLES,
        ),
        heat=[
            "<resources>/automatic/{shapes}/{scenario}/documents/heat_allocated_potential.tif"
        ],
        contested_heat=[
            "<resources>/automatic/{shapes}/{scenario}/documents/contested_heat_potential.tif"
        ],
        heat_potential_fig=report_output(
            "<results>/{shapes}/{scenario}/figures/heat_potential_overview.pdf",
            category="heat",
            subcategory="{shapes}/{scenario}",
            labels={"Figure": "Heat supplied to networks"},
        ),
        heat_lcoh_fig=report_output(
            "<results>/{shapes}/{scenario}/figures/lcoh_overview.pdf",
            category="heat",
            subcategory="{shapes}/{scenario}",
            labels={"Figure": "LCOH raster"},
        ),
        regional_heat_fig=report_output(
            "<results>/{shapes}/{scenario}/figures/regional_heat_potential.pdf",
            category="heat",
            subcategory="{shapes}/{scenario}",
            labels={"Figure": "Regional heating potential"},
        ),
    )
