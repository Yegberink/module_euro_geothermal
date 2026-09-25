"""Configuration validation and model settings for the geothermal module."""

from typing import Any

MODEL_KEYS = (
    "carrier_out",
    "max_depth",
    "include_egs",
    "EGS_cutoff",
    "SED_NAN_POLICY",
    "drilling_cost_curve",
    "drilling_cost_curve_definitions",
    "stimulation_cost_per_doublet",
    "pump_cost",
    "lifetime",
    "capacity_factor",
    "fixed_OM_rate",
    "parasitic_fraction",
    "orc_money_per_kw",
    "flash_money_per_kw",
    "kalina_money_per_kw",
    "well_spacing",
    "flowrate_egs",
    "flowrate_conventional",
    "min_dt",
    "temp_turbine_out",
)


def model_parameters(cfg: dict[str, Any]) -> dict[str, Any]:
    """Select calculation settings tracked by Snakemake for reruns."""
    return {key: cfg[key] for key in MODEL_KEYS}


def validate_module_config(cfg: dict[str, Any]) -> None:
    """Validate configuration relationships not expressible in JSON Schema."""
    if cfg["carrier_out"] != "power":
        raise ValueError("Only carrier_out='power' is supported.")

    curve = cfg["drilling_cost_curve"]
    if curve not in cfg["drilling_cost_curve_definitions"]:
        raise ValueError(f"drilling_cost_curve {curve!r} has no matching definition.")

    _validate_depth_categories(cfg.get("depth_categories", []))
    _validate_fixed_bins(cfg.get("fixed_bins", []))


def _validate_depth_categories(categories: list[list[float]]) -> None:
    """Require ordered, non-overlapping depth intervals within the source model."""
    previous_stop = 0.0
    for category in categories:
        start, stop = map(float, category)
        if start < 0 or stop > 10_000 or start >= stop:
            raise ValueError(
                f"Invalid depth category {category!r}; expected 0 <= start < stop <= 10000."
            )
        if start < previous_stop:
            raise ValueError("depth_categories must be ordered and non-overlapping.")
        previous_stop = stop


def _validate_fixed_bins(bins: list[str]) -> None:
    """Require ordered, non-overlapping LCOE intervals."""
    previous_stop = 0.0
    for value in bins:
        start_text, stop_text = value.split("-", maxsplit=1)
        start, stop = float(start_text), float(stop_text)
        if start < 0 or start >= stop:
            raise ValueError(f"Invalid fixed LCOE bin {value!r}.")
        if start < previous_stop:
            raise ValueError("fixed_bins must be ordered and non-overlapping.")
        previous_stop = stop
